"""Build the local two-zoom multi-grid (fine = the grid ICON runs on, e.g.
R2B4 or R2B8; coarse = one refinement level up, e.g. R2B3 or R2B7) for online
FieldSpaceNN training, directly from COMIN's native ICON grid data plus two
static grid files read once at startup (never per-timestep).

Design summary
--------------
FieldSpaceNN's `MG_Transformer`/`GridLayer` expects each zoom level as a
dense `(adjc, adjc_mask, coords)` triple: `adjc`/`adjc_mask` are `(n, 9)`
(column 0 = self, columns 1-8 = up to 8 directional neighbors in a
HEALPix-shaped convention; ICON triangular cells only have 3 real
neighbors, tiled into the 8 slots -- see
`fieldspacenn...icon_neighbor_cell_index_to_adjc`).

Two zooms are built:

* Fine zoom (zoom index 1): the rank's own local ICON native-grid patch
  (owned cells + halo), same node space as `graph_utils.build_local_graph`.
  Unlike FieldSpaceNN's disk-based `icon_neighbor_cell_index_to_adjc`
  (which assumes a *closed* global grid -- every cell has exactly 3 real
  neighbors), a rank's local patch is not closed: a boundary halo cell can
  be missing 1-2 of its 3 neighbors (not shipped to this rank at all). So
  the fine-zoom adjacency here has real missing-neighbor handling: any
  unavailable neighbor slot falls back to self, with `adjc_mask=True` at
  that slot -- the same "duplicate -> self, mask it" idiom
  `healpix_get_adjacent_cell_indices` already uses for HEALPix's polar
  gaps.
* Coarse zoom (zoom index 0): built from a **static coarse grid file, read
  once** at plugin startup -- this is small, fixed grid *topology*, the
  same category of one-time disk read `icon_grid_to_mgrid` already does
  offline, not the "no disk I/O for per-timestep training data" concern
  that motivates avoiding YAC/HEALPix interpolation elsewhere in this
  project.

The fine-to-coarse parent-child relationship is **not** derived by index
arithmetic or nearest-neighbor geometry -- both were tried and found
unreliable against the real grid files this project uses (see the plan
document written for this task). Instead, ICON's own `parent_cell_index`
variable (present in the *fine* grid file, populated by the grid generator
with the exact global parent index per global fine cell) is used directly,
looked up via COMIN's `domain.cells.glb_index`. A rank's local fine cell
only contributes to a coarse "token" for parent groups where *all 4* of
that parent's children are present in this rank's local (owned + halo)
cell set this step -- incomplete groups (a real, expected consequence of
ICON's domain decomposition not respecting R2Bn quad-tree nesting) are
simply excluded, not approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr

from fieldspacenn.src.modules.grids.grid_utils import icon_neighbor_cell_index_to_adjc

# Callers (fieldspace_plugin.py, and MEssE/tests/conftest.py for unit tests)
# already put the project root on sys.path before importing this module --
# see their own _PLUGIN_DIR/_PROJECT_ROOT bootstrap -- so a plain dotted
# import is safe here without repeating that bootstrap (same approach
# graph_utils.py itself uses for this same import).
from MEssE.utils.icon_online_helper import flat_index

# The 3 real ICON neighbors tiled into GridLayer's 8 non-self columns,
# matching fieldspacenn.grid_utils.icon_neighbor_cell_index_to_adjc's
# convention exactly.
_TILE_PATTERN = [0, 1, 2, 0, 1, 2, 0, 1]

# FieldSpaceNN's GridLayer.__init__ (not modified -- see build_local_mgrid)
# runs a HEALPix "polar discontinuity" fixup (`propagate_assignments`) at
# exactly these two adjacency columns, for any zoom at list-position >= 1
# (always our fine zoom). That fixup assumes a regular, HEALPix-like
# reverse-neighbor structure and crashes (RuntimeError: stack expects each
# tensor to be equal size) on real, boundary-clipped ICON local patches --
# confirmed on a live run (job 27394205), not a toy-test artifact.
_GRIDLAYER_POLAR_FIX_COLUMNS = (2, 6)


@dataclass
class FineAdjacency:
    """Local fine-zoom adjacency, in the same flat node-id space as
    `graph_utils.LocalGraph` (owned + halo + unused padding)."""

    n_nodes: int
    n_valid: int
    adjc: np.ndarray  # (n_nodes, 9) int64, local flat ids
    adjc_mask: np.ndarray  # (n_nodes, 9) bool
    owned_mask: np.ndarray  # (n_nodes,) bool
    clon: np.ndarray  # (n_nodes,) float64, radians
    clat: np.ndarray  # (n_nodes,) float64, radians


@dataclass
class CoarseGrid:
    """Global coarse-zoom topology. Built by :func:`load_coarse_grid` from a
    static grid file, or directly in-memory for tests (bypassing the file
    read entirely)."""

    n_cells: int
    adjc: np.ndarray  # (n_cells, 9) int64, global 0-based ids
    adjc_mask: np.ndarray  # (n_cells, 9) bool


@dataclass
class LocalMGrid:
    """The compact, model-ready two-zoom multi-grid for one rank. Static for
    the whole run (cell *values* still come from COMIN each step)."""

    # Full local fine space (needed for input extraction / write-back bookkeeping).
    n_nodes_fine: int
    n_valid_fine: int
    fine_owned_mask: torch.Tensor  # (n_nodes_fine,) bool

    # Compact subset actually fed to the model: fine_local_ids_kept[4t:4t+4]
    # are exactly the 4 children of coarse token t (order is a stable,
    # deterministic sort of local ids -- not otherwise physically meaningful).
    fine_local_ids_kept: torch.Tensor  # (4*n_coarse,) int64, into full fine space
    fine_adjc: torch.Tensor  # (4*n_coarse, 9) int64, local compact ids
    fine_adjc_mask: torch.Tensor  # (4*n_coarse, 9) bool
    fine_coords: torch.Tensor  # (4*n_coarse, 2) float32 [lon, lat] radians
    fine_compact_owned_mask: torch.Tensor  # (4*n_coarse,) bool

    n_coarse: int
    coarse_adjc: torch.Tensor  # (n_coarse, 9) int64, local coarse token ids
    coarse_adjc_mask: torch.Tensor  # (n_coarse, 9) bool
    coarse_coords: torch.Tensor  # (n_coarse, 2) float32, mean of the 4 children

    n_fine_kept: int  # == 4 * n_coarse
    parent_group_size_histogram: Dict[int, int] = field(default_factory=dict)


def _local_cell_basics(domain, nproma: int):
    """The handful of per-cell fields every local-grid builder in this
    project needs (n_nodes, n_valid, owned_mask, clon, clat), computed
    directly from COMIN's `domain.cells`. Deliberately not shared with
    `graph_utils.build_local_graph` (which duplicates this same
    computation for its own edge-list construction) -- kept as two
    independent, self-contained modules rather than have one import the
    other for a handful of fields.
    """
    n_valid = int(domain.cells.ncells)
    n_nodes = int(domain.cells.nblks) * nproma

    # order="F": domain.cells.* arrays are (nproma, nblks) with idx (nproma)
    # the fast-varying axis, matching flat_index's own convention
    # (c = (blk-1)*nproma + (idx-1)) -- the default order="C" would instead
    # vary blk fastest, silently permuting which physical cell ends up at
    # flat position c whenever nblks > 1 (a real bug, though currently
    # dormant: every GPU run so far uses nblocks_c=1, under which "C" and
    # "F" order coincide since there's only one block to begin with).
    clon_full = np.ravel(np.asarray(domain.cells.clon), order="F").astype(np.float64)
    clat_full = np.ravel(np.asarray(domain.cells.clat), order="F").astype(np.float64)
    # decomp_domain == 0 marks prognostic ("owned") cells; > 0 marks halo
    # cells at increasing distance from the owned region; < 0 is unused
    # padding at the end of the last block.
    decomp = np.ravel(np.asarray(domain.cells.decomp_domain), order="F").astype(np.int64)
    owned = decomp == 0

    return n_nodes, n_valid, owned, clon_full, clat_full


def build_fine_adjacency(domain, nproma: int) -> FineAdjacency:
    """Build the local fine-zoom (adjc, adjc_mask) directly from COMIN's
    domain.cells.neighbor_idx/neighbor_blk, with real missing-neighbor
    handling (a rank's local patch is not a closed grid, unlike the
    disk-based icon_neighbor_cell_index_to_adjc's assumption).
    """
    n_nodes, n_valid, owned, clon_full, clat_full = _local_cell_basics(domain, nproma)

    neighbor_idx = np.asarray(domain.cells.neighbor_idx)  # (nproma, nblks_c, 3)
    neighbor_blk = np.asarray(domain.cells.neighbor_blk)
    n_nbr = neighbor_idx.shape[-1]
    assert n_nbr == 3, f"expected 3 ICON neighbors per cell, got {n_nbr}"

    nproma_ids, blk_ids = np.meshgrid(
        np.arange(1, nproma + 1), np.arange(1, neighbor_idx.shape[1] + 1), indexing="ij"
    )
    # order="F" here too, for the same reason as _local_cell_basics above:
    # these reshapes must put flat position c at (idx=c%nproma, blk=c//nproma)
    # to agree with clon_full/clat_full/owned's own flattening and with
    # extract_icon_cells' Fortran-order convention elsewhere in this
    # project -- the default "C" order would instead vary blk fastest,
    # silently misaligning adjc's rows with every other array indexed by
    # flat cell id whenever nblks_c > 1 (verified by hand-checking the
    # index arithmetic; dormant today only because nblocks_c=1 on every
    # real GPU run so far).
    self_flat = flat_index(nproma_ids, blk_ids, nproma).reshape(n_nodes, order="F")
    nbr_flat = flat_index(neighbor_idx, neighbor_blk, nproma).reshape(n_nodes, n_nbr, order="F")

    valid = (
        (neighbor_idx.reshape(n_nodes, n_nbr, order="F") > 0)
        & (neighbor_blk.reshape(n_nodes, n_nbr, order="F") > 0)
        & (nbr_flat >= 0)
        & (nbr_flat < n_nodes)
    )
    nbr_safe = np.where(valid, nbr_flat, self_flat[:, None])  # invalid slot -> self

    tiled_nbr = nbr_safe[:, _TILE_PATTERN]  # (n_nodes, 8)
    tiled_valid = valid[:, _TILE_PATTERN]  # (n_nodes, 8)

    adjc = np.concatenate([self_flat[:, None], tiled_nbr], axis=1)  # (n_nodes, 9)
    adjc_mask = np.concatenate(
        [np.zeros((n_nodes, 1), dtype=bool), ~tiled_valid], axis=1
    )  # mask=True: no real neighbor here, fell back to self

    return FineAdjacency(
        n_nodes=n_nodes,
        n_valid=n_valid,
        adjc=adjc.astype(np.int64),
        adjc_mask=adjc_mask,
        owned_mask=owned,
        clon=clon_full,
        clat=clat_full,
    )


def load_coarse_grid(coarse_grid_path: str) -> CoarseGrid:
    """Read a static, closed-global coarse grid file once (at plugin
    startup) and build its (adjc, adjc_mask) via FieldSpaceNN's own
    disk-based icon_neighbor_cell_index_to_adjc -- valid here because a full
    global grid file genuinely has exactly 3 real neighbors per cell, unlike
    a rank's local (possibly boundary-clipped) patch.
    """
    ds = xr.open_dataset(coarse_grid_path)
    adjc_t, adjc_mask_t = icon_neighbor_cell_index_to_adjc(ds)
    return CoarseGrid(
        n_cells=int(adjc_t.shape[0]),
        adjc=adjc_t.numpy().astype(np.int64),
        adjc_mask=adjc_mask_t.numpy(),
    )


def load_parent_index(fine_grid_path: str) -> np.ndarray:
    """Read the fine grid file's own `parent_cell_index` once: the exact,
    grid-generator-assigned global coarse parent for every global fine cell
    (1-based in the file; returned 0-based).
    """
    ds = xr.open_dataset(fine_grid_path)
    return ds["parent_cell_index"].values.astype(np.int64) - 1


def check_grid_files(
    parent_index_global: np.ndarray, coarse_grid: CoarseGrid, ncells_global: int
) -> None:
    """Raise ``ValueError`` unless the fine grid file describes the grid ICON
    runs on (``ncells_global``, COMIN's ``domain.cells.ncells_global``) and
    the coarse grid file is its parent grid, one refinement level up.

    Without this, a file pair for another resolution fails later with an
    IndexError inside :func:`build_local_mgrid`, or trains on a scrambled
    grid if the cell counts happen to fit.
    """
    n_fine = int(parent_index_global.shape[0])
    if n_fine != int(ncells_global):
        raise ValueError(
            f"fine grid file has {n_fine} cells but ICON runs on {ncells_global}: "
            "it must be the grid file of the ICON run itself"
        )
    if 4 * coarse_grid.n_cells != n_fine:
        raise ValueError(
            f"coarse grid file has {coarse_grid.n_cells} cells, expected {n_fine // 4} "
            f"(one refinement level above the {n_fine}-cell fine grid)"
        )
    if parent_index_global.min() < 0 or parent_index_global.max() >= coarse_grid.n_cells:
        raise ValueError(
            f"fine grid parent_cell_index spans [{parent_index_global.min() + 1}, "
            f"{parent_index_global.max() + 1}], outside the coarse grid's "
            f"{coarse_grid.n_cells} cells"
        )


def build_local_mgrid(
    domain,
    nproma: int,
    coarse_grid: CoarseGrid,
    parent_index_global: np.ndarray,
    device: Optional[torch.device] = None,
) -> LocalMGrid:
    """Build this rank's compact two-zoom multi-grid.

    Only coarse parent groups whose all 4 children are present in this
    rank's local (owned + halo) fine cell set are kept as coarse tokens;
    the corresponding 4 fine children are the only fine cells fed to the
    model. Everything else (orphan fine cells belonging to an incomplete
    group) is excluded, not masked-and-kept -- see module docstring.
    """
    fine = build_fine_adjacency(domain, nproma)

    # domain.cells.glb_index: 1-based global index per local (owned+halo)
    # cell, length == fine.n_valid, in the same local ordering as the first
    # fine.n_valid flat ids (see extract_icon_cells/insert_icon_cells' same
    # assumption elsewhere in this codebase).
    glb_index = np.ravel(np.asarray(domain.cells.glb_index)).astype(np.int64)
    n_valid = fine.n_valid
    local_ids = np.arange(n_valid, dtype=np.int64)
    parent_global = parent_index_global[glb_index[:n_valid] - 1]

    order = np.argsort(parent_global, kind="stable")
    sorted_parents = parent_global[order]
    sorted_local_ids = local_ids[order]
    unique_parents, group_start, group_count = np.unique(
        sorted_parents, return_index=True, return_counts=True
    )

    histogram: Dict[int, int] = {}
    for c in group_count.tolist():
        histogram[c] = histogram.get(c, 0) + 1

    complete = group_count == 4
    kept_parents = unique_parents[complete]
    kept_starts = group_start[complete]
    n_coarse = int(kept_parents.shape[0])

    fine_local_ids_kept = np.empty(4 * n_coarse, dtype=np.int64)
    for t, start in enumerate(kept_starts.tolist()):
        fine_local_ids_kept[4 * t : 4 * t + 4] = np.sort(sorted_local_ids[start : start + 4])

    # --- coarse adjacency, re-indexed to the local compact token space ---
    parent_to_compact = {int(p): t for t, p in enumerate(kept_parents.tolist())}
    coarse_adjc = np.zeros((n_coarse, 9), dtype=np.int64)
    coarse_adjc_mask = np.zeros((n_coarse, 9), dtype=bool)
    for t, p in enumerate(kept_parents.tolist()):
        for col in range(9):
            g = int(coarse_grid.adjc[p, col])
            if not coarse_grid.adjc_mask[p, col] and g in parent_to_compact:
                coarse_adjc[t, col] = parent_to_compact[g]
                coarse_adjc_mask[t, col] = False
            else:
                coarse_adjc[t, col] = t  # self
                coarse_adjc_mask[t, col] = True

    # --- fine adjacency, re-indexed to the compact 4*n_coarse space ---
    full_to_compact = -np.ones(fine.n_nodes, dtype=np.int64)
    full_to_compact[fine_local_ids_kept] = np.arange(4 * n_coarse, dtype=np.int64)

    fine_adjc_full = fine.adjc[fine_local_ids_kept]  # (4*n_coarse, 9), full-space ids
    fine_adjc_mask_full = fine.adjc_mask[fine_local_ids_kept]

    mapped = full_to_compact[fine_adjc_full]  # (4*n_coarse, 9), -1 if not kept
    self_compact = np.arange(4 * n_coarse, dtype=np.int64)[:, None]
    fine_adjc = np.where(mapped >= 0, mapped, self_compact)
    fine_adjc_mask = fine_adjc_mask_full | (mapped < 0)

    # Never mark _GRIDLAYER_POLAR_FIX_COLUMNS as masked: GridLayer's
    # propagate_assignments crashes on real per-rank patches at those two
    # columns specifically (see the constant's docstring). Since `fine_adjc`
    # already falls back to self wherever the true neighbor was missing
    # (the `np.where(mapped >= 0, mapped, self_compact)` above), forcing
    # the mask to False there just means "treat that self-fallback as
    # valid" for those two columns -- self is always a legitimate,
    # in-bounds cell, so this is a safe (if slightly lossy: those two
    # columns lose their "ignore me" signal for the minority of cells
    # whose true neighbor there was genuinely missing) substitute. The
    # other tile copies of the same logical neighbor slot (columns
    # 1, 3, 4, 5, 7, 8) keep their correct mask, so the missing-neighbor
    # information is not lost entirely, just redundantly under-represented
    # at these two columns.
    fine_adjc_mask[:, list(_GRIDLAYER_POLAR_FIX_COLUMNS)] = False

    fine_coords_kept = np.stack(
        [fine.clon[fine_local_ids_kept], fine.clat[fine_local_ids_kept]], axis=-1
    )  # (4*n_coarse, 2)
    coarse_coords = fine_coords_kept.reshape(n_coarse, 4, 2).mean(axis=1)  # (n_coarse, 2)

    fine_compact_owned = fine.owned_mask[fine_local_ids_kept]  # (4*n_coarse,)

    def _t(arr, dtype):
        tt = torch.as_tensor(arr, dtype=dtype)
        return tt.to(device) if device is not None else tt

    return LocalMGrid(
        n_nodes_fine=fine.n_nodes,
        n_valid_fine=fine.n_valid,
        fine_owned_mask=_t(fine.owned_mask, torch.bool),
        fine_local_ids_kept=_t(fine_local_ids_kept, torch.int64),
        fine_adjc=_t(fine_adjc, torch.int64),
        fine_adjc_mask=_t(fine_adjc_mask, torch.bool),
        fine_coords=_t(fine_coords_kept, torch.float32),
        fine_compact_owned_mask=_t(fine_compact_owned, torch.bool),
        n_coarse=n_coarse,
        coarse_adjc=_t(coarse_adjc, torch.int64),
        coarse_adjc_mask=_t(coarse_adjc_mask, torch.bool),
        coarse_coords=_t(coarse_coords, torch.float32),
        n_fine_kept=4 * n_coarse,
        parent_group_size_histogram=histogram,
    )


def pool_fine_to_coarse(fine_x_compact: torch.Tensor, n_coarse: int) -> torch.Tensor:
    """Mean-pool the compact fine-zoom tensor into the coarse zoom.

    Cheap: a plain reshape + mean, since `LocalMGrid.fine_local_ids_kept`
    already guarantees each coarse token's 4 children occupy a contiguous
    group in the compact ordering.
    """
    return fine_x_compact.view(n_coarse, 4, -1).mean(dim=1)
