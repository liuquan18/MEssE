"""Nested multi-level ICON grids for online FieldSpaceNN training, with every
coarse backbone cell assigned to exactly one GPU rank.

Levels
------
A model on, e.g., an R2B4 backbone plus R2B6, R2B7 and R2B8 residuals needs a
FieldSpaceNN grid layer for *every* level from the backbone to the fine grid
(R2B4, R2B5, ..., R2B8): `MG_base_model` keys grid layers by list position,
and zoom ``z`` must be refined ``4**z`` times relative to zoom 0. Zoom ``z``
here is ``level - base_level``.

The grid files must be nested by index: the parent of cell ``c`` (0-based)
on level ``L`` is cell ``c // 4`` on level ``L - 1``. Then the descendants of
backbone cell ``b`` on level ``base + k`` are the contiguous range
``b * 4**k .. (b + 1) * 4**k - 1``, which is the nested ordering
FieldSpaceNN's tokenizer and `encode_zooms` assume. The MPI-M grids
(``/pool/data/ICON/grids/mpim/Earth_IcosS_*.nc``, same cells as
``public/mpim/00xx/icon_grid_00xx_R02Bnn_G.nc``) are nested this way.
:func:`load_level_grids` checks it from each file's ``parent_cell_index``.

Rank patches
------------
ICON's domain decomposition ignores the grid hierarchy, so the descendants
of a backbone cell are usually spread over several ranks, and the 2-row halo
is far too narrow to complete them. Instead each backbone cell is assigned to
the GPU rank that owns most of its fine cells (:func:`assign_backbone_owners`).
A rank's model patch is the full descendant tree of its backbone cells
(:func:`build_nested_mgrid`), and :class:`CellExchange` moves fine values
between ranks with ``Alltoallv``: owned cells to the patch that holds them,
and predictions back to the owning rank for write-back. Every fine cell is in
exactly one patch, so each is trained on and predicted exactly once. Halo
cells are not used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import netCDF4
import numpy as np
import torch

# MPI-M grid files per R2B level (see /pool/data/ICON/grids/mpim/grid_notes.txt).
MPIM_GRID_FILES = {
    3: "Earth_IcosS_0320km.nc",
    4: "Earth_IcosS_0160km.nc",
    5: "Earth_IcosS_0080km.nc",
    6: "Earth_IcosS_0040km.nc",
    7: "Earth_IcosS_0020km.nc",
    8: "Earth_IcosS_0010km.nc",
    9: "Earth_IcosS_0005km.nc",
}

# The 3 real ICON neighbors tiled into GridLayer's 8 non-self columns, as in
# icon_mgrid_utils and fieldspacenn's icon_neighbor_cell_index_to_adjc.
_TILE_PATTERN = [0, 1, 2, 0, 1, 2, 0, 1]
# GridLayer's HEALPix polar fixup crashes on masked entries in these columns
# (see icon_mgrid_utils._GRIDLAYER_POLAR_FIX_COLUMNS); they are never masked.
_GRIDLAYER_POLAR_FIX_COLUMNS = (2, 6)
# GridLayer writes column indices up to 8 into masked adjacency slots, so
# every level needs at least 9 cells: at least 9 backbone cells per patch.
MIN_BACKBONE_CELLS = 9


# ----------------------------------------------------------------------------
# Global grids
# ----------------------------------------------------------------------------


@dataclass
class LevelGrid:
    """Global topology and cell centers of one refinement level."""

    level: int
    neighbors: np.ndarray  # (n_cells, 3) int64, 0-based global ids, -1 where none
    lon: np.ndarray  # (n_cells,) float32, radians
    lat: np.ndarray  # (n_cells,) float32, radians

    @property
    def n_cells(self) -> int:
        return int(self.lon.shape[0])


def mpim_grid_paths(grid_dir: str, base_level: int, fine_level: int) -> Dict[int, str]:
    """MPI-M grid file for every level from ``base_level`` to ``fine_level``."""
    missing = [lvl for lvl in range(base_level, fine_level + 1) if lvl not in MPIM_GRID_FILES]
    if missing:
        raise ValueError(f"no MPI-M grid file known for R2B levels {missing}")
    return {lvl: os.path.join(grid_dir, MPIM_GRID_FILES[lvl]) for lvl in range(base_level, fine_level + 1)}


def check_index_nesting(parent_cell_index: np.ndarray, n_parent_cells: int, level: int) -> None:
    """Raise ``ValueError`` unless every cell ``c`` (0-based) of ``level`` has
    parent ``c // 4`` (``parent_cell_index`` is 1-based, as in grid files)."""
    n_cells = int(parent_cell_index.shape[0])
    if n_cells != 4 * n_parent_cells:
        raise ValueError(
            f"R2B{level} grid has {n_cells} cells, expected 4 x {n_parent_cells} (R2B{level - 1})"
        )
    if not np.array_equal(np.asarray(parent_cell_index, dtype=np.int64), np.arange(n_cells) // 4 + 1):
        raise ValueError(
            f"R2B{level} grid is not index-nested (parent_cell_index != cell // 4 + 1); "
            "use a nested grid family such as /pool/data/ICON/grids/mpim/Earth_IcosS_*"
        )


def _read_var(ds, *names: str) -> np.ndarray:
    for name in names:
        if name in ds.variables:
            return np.asarray(ds.variables[name][:])
    raise KeyError(f"none of {names} in {ds.filepath()}")


def load_level_grids(
    grid_paths: Mapping[int, str], ncells_global: Optional[int] = None
) -> Dict[int, LevelGrid]:
    """Read every level's neighbors and cell centers once, checking that the
    levels are consecutive and index-nested, and (if given) that the finest
    level is the grid ICON runs on (``ncells_global`` cells)."""
    levels = sorted(grid_paths)
    if levels != list(range(levels[0], levels[-1] + 1)):
        raise ValueError(f"need a grid file for every level from R2B{levels[0]} to R2B{levels[-1]}")
    grids: Dict[int, LevelGrid] = {}
    for level in levels:
        with netCDF4.Dataset(grid_paths[level]) as ds:
            ds.set_auto_mask(False)
            neighbors = _read_var(ds, "neighbor_cell_index").T.astype(np.int64) - 1
            lon = _read_var(ds, "clon", "lon_cell_centre").astype(np.float32)
            lat = _read_var(ds, "clat", "lat_cell_centre").astype(np.float32)
            if level > levels[0]:
                check_index_nesting(_read_var(ds, "parent_cell_index"), grids[level - 1].n_cells, level)
        grids[level] = LevelGrid(level=level, neighbors=neighbors, lon=lon, lat=lat)
    fine = grids[levels[-1]]
    if ncells_global is not None and fine.n_cells != int(ncells_global):
        raise ValueError(
            f"R2B{levels[-1]} grid file has {fine.n_cells} cells but ICON runs on {ncells_global}"
        )
    return grids


def check_cell_centers(
    grid: LevelGrid, glb: np.ndarray, lon: np.ndarray, lat: np.ndarray, tol: float = 1e-4
) -> None:
    """Raise ``ValueError`` unless the cell centers ICON reports (radians)
    for the 0-based global ids ``glb`` lie within ``tol`` radians (about
    0.6 km) of the grid file's. Catches a grid file with the same cell count
    but a different numbering, and a wrong local-to-global mapping."""
    glb = np.asarray(glb, dtype=np.int64)
    lat = np.asarray(lat, dtype=np.float64)
    dlon = (np.asarray(lon, dtype=np.float64) - grid.lon[glb] + np.pi) % (2 * np.pi) - np.pi
    dist = np.hypot(lat - grid.lat[glb], np.cos(lat) * dlon)
    if dist.size and dist.max() > tol:
        raise ValueError(
            f"{int((dist > tol).sum())} of {dist.size} cell centers differ from the R2B{grid.level} "
            f"grid file by more than {tol} rad (max {dist.max():.3g}): wrong grid file or cell numbering"
        )


# ----------------------------------------------------------------------------
# Backbone ownership
# ----------------------------------------------------------------------------


def backbone_counts(owned_glb: np.ndarray, n_backbone: int, refinement: int) -> np.ndarray:
    """Owned fine cells per backbone cell, (n_backbone,) int64. ``owned_glb``
    are this rank's owned fine cells (0-based global ids), ``refinement`` is
    the number of fine cells per backbone cell (``4**(fine - base)``)."""
    return np.bincount(np.asarray(owned_glb, dtype=np.int64) // refinement, minlength=n_backbone)


def assign_backbone_owners(counts: np.ndarray, refinement: int) -> np.ndarray:
    """Owner rank per backbone cell from all ranks' :func:`backbone_counts`
    stacked as ``(n_ranks, n_backbone)``: the rank owning most of its fine
    cells, the lowest such rank on ties.

    Raises unless the counts add up to exactly ``refinement`` per backbone
    cell, i.e. unless every fine cell is owned by exactly one rank.
    """
    counts = np.asarray(counts)
    totals = counts.sum(axis=0)
    if not np.all(totals == refinement):
        bad = int(np.flatnonzero(totals != refinement)[0])
        raise ValueError(
            f"backbone cell {bad} has {int(totals[bad])} owned fine cells over all ranks, "
            f"expected {refinement}: the fine cells are not partitioned over these ranks"
        )
    return np.argmax(counts, axis=0)


# ----------------------------------------------------------------------------
# Rank patch
# ----------------------------------------------------------------------------


def patch_cell_ids(backbone_ids: np.ndarray, k: int) -> np.ndarray:
    """Global ids, in nested order, of all level-``base + k`` descendants of
    the (sorted) backbone cells."""
    n = 4**k
    return (np.asarray(backbone_ids, dtype=np.int64)[:, None] * n + np.arange(n)).ravel()


@dataclass
class NestedMGrid:
    """One rank's model patch on every level from ``base_level`` to
    ``fine_level``; entry ``z`` of each list is zoom ``z`` (level
    ``base_level + z``). Cells are in nested order: the ``4**k`` level
    ``base + k`` cells of backbone cell ``t`` are rows ``t*4**k .. (t+1)*4**k - 1``.
    Static for the whole run."""

    base_level: int
    fine_level: int
    backbone_ids: np.ndarray  # (n_backbone,) sorted 0-based global backbone ids
    adjc: List[torch.Tensor]  # per zoom, (n_z, 9) int64 local ids
    adjc_mask: List[torch.Tensor]  # per zoom, (n_z, 9) bool, True = no neighbor there
    coords: List[torch.Tensor]  # per zoom, (n_z, 2) float32 [lon, lat] radians

    @property
    def n_backbone(self) -> int:
        return int(self.backbone_ids.shape[0])

    @property
    def refinement(self) -> int:
        return 4 ** (self.fine_level - self.base_level)

    @property
    def n_fine(self) -> int:
        return self.n_backbone * self.refinement

    def zoom(self, level: int) -> int:
        return level - self.base_level

    def n_cells(self, level: int) -> int:
        return self.n_backbone * 4 ** self.zoom(level)

    def fieldspace_mgrids(self) -> List[Dict[str, object]]:
        """FieldSpaceNN's ``mgrids`` list. Copies: GridLayer edits adjacency
        in place."""
        return [
            {"coords": c.clone(), "adjc": a.clone(), "adjc_mask": m.clone(), "zoom": z}
            for z, (a, m, c) in enumerate(zip(self.adjc, self.adjc_mask, self.coords))
        ]


def build_nested_mgrid(
    grids: Mapping[int, LevelGrid],
    backbone_ids: np.ndarray,
    device: Optional[torch.device] = None,
) -> NestedMGrid:
    """The patch of the given backbone cells on every level in ``grids``.

    Neighbors outside the patch fall back to the cell itself and are masked,
    except in the GridLayer polar-fix columns (see module constants).
    """
    levels = sorted(grids)
    backbone_ids = np.sort(np.asarray(backbone_ids, dtype=np.int64))
    adjc, adjc_mask, coords = [], [], []
    for level in levels:
        grid = grids[level]
        ids = patch_cell_ids(backbone_ids, level - levels[0])  # sorted
        nbr = grid.neighbors[ids]
        pos = np.minimum(np.searchsorted(ids, nbr), ids.shape[0] - 1)
        valid = (nbr >= 0) & (ids[pos] == nbr)
        self_ids = np.arange(ids.shape[0], dtype=np.int64)[:, None]
        local = np.where(valid, pos, self_ids)

        a = np.concatenate([self_ids, local[:, _TILE_PATTERN]], axis=1)
        m = np.concatenate([np.zeros((ids.shape[0], 1), dtype=bool), ~valid[:, _TILE_PATTERN]], axis=1)
        m[:, list(_GRIDLAYER_POLAR_FIX_COLUMNS)] = False
        c = np.stack([grid.lon[ids], grid.lat[ids]], axis=-1)

        adjc.append(torch.as_tensor(a, dtype=torch.int64, device=device))
        adjc_mask.append(torch.as_tensor(m, dtype=torch.bool, device=device))
        coords.append(torch.as_tensor(c, dtype=torch.float32, device=device))
    return NestedMGrid(
        base_level=levels[0],
        fine_level=levels[-1],
        backbone_ids=backbone_ids,
        adjc=adjc,
        adjc_mask=adjc_mask,
        coords=coords,
    )


def patch_positions(glb: np.ndarray, backbone_ids: np.ndarray, refinement: int) -> np.ndarray:
    """Row of each fine cell (0-based global id) in the nested fine patch of
    ``backbone_ids`` (sorted). Raises if a cell is not in the patch."""
    glb = np.asarray(glb, dtype=np.int64)
    backbone = glb // refinement
    idx = np.searchsorted(backbone_ids, backbone)
    inside = idx < backbone_ids.shape[0]
    inside[inside] = backbone_ids[idx[inside]] == backbone[inside]
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} fine cells are not in this rank's patch")
    return idx * refinement + glb % refinement


# ----------------------------------------------------------------------------
# Cell exchange between ranks
# ----------------------------------------------------------------------------


def _displacements(counts: np.ndarray) -> List[int]:
    return [0] + np.cumsum(counts)[:-1].tolist()


@dataclass
class CellExchange:
    """Precomputed routing of fine-cell values between each rank's owned
    ICON cells and the model patches. All ranks of ``comm`` must call
    :meth:`to_patch` and :meth:`from_patch` together (they are collective)."""

    send_local_ids: np.ndarray  # owned local flat cell ids, grouped by destination rank
    send_counts: np.ndarray  # (n_ranks,) cells sent to each rank
    recv_counts: np.ndarray  # (n_ranks,) cells received from each rank
    recv_patch_pos: np.ndarray  # patch row of every received cell, in receive order
    n_patch: int

    @classmethod
    def build(
        cls,
        comm,
        owned_local: np.ndarray,
        owned_glb: np.ndarray,
        owner: np.ndarray,
        backbone_ids: np.ndarray,
        refinement: int,
    ) -> "CellExchange":
        """Collective. ``owned_local``/``owned_glb``: this rank's owned fine
        cells as local flat ids and 0-based global ids; ``owner``: owner rank
        per backbone cell (:func:`assign_backbone_owners`); ``backbone_ids``:
        this rank's backbone cells."""
        n_ranks = comm.Get_size()
        owned_local = np.asarray(owned_local, dtype=np.int64)
        owned_glb = np.asarray(owned_glb, dtype=np.int64)
        dest = np.asarray(owner)[owned_glb // refinement]
        order = np.lexsort((owned_glb, dest))
        send_glb = np.ascontiguousarray(owned_glb[order])
        send_counts = np.bincount(dest, minlength=n_ranks).astype(np.int64)

        recv_counts = np.empty(n_ranks, dtype=np.int64)
        comm.Alltoall(send_counts, recv_counts)
        recv_glb = np.empty(int(recv_counts.sum()), dtype=np.int64)
        comm.Alltoallv(
            [send_glb, (send_counts.tolist(), _displacements(send_counts))],
            [recv_glb, (recv_counts.tolist(), _displacements(recv_counts))],
        )

        backbone_ids = np.sort(np.asarray(backbone_ids, dtype=np.int64))
        n_patch = backbone_ids.shape[0] * refinement
        pos = patch_positions(recv_glb, backbone_ids, refinement)
        if pos.shape[0] != n_patch or not np.array_equal(np.sort(pos), np.arange(n_patch)):
            raise ValueError(
                f"received {pos.shape[0]} fine cells for a patch of {n_patch}: "
                "cells are missing or duplicated"
            )
        return cls(
            send_local_ids=owned_local[order],
            send_counts=send_counts,
            recv_counts=recv_counts,
            recv_patch_pos=pos,
            n_patch=n_patch,
        )

    def to_patch(self, comm, local_values: np.ndarray) -> np.ndarray:
        """Collective. ``local_values``: (n_local_cells, nlev) this rank's
        ICON cells. Returns (n_patch, nlev) float32 in nested patch order."""
        nlev = local_values.shape[1]
        send = np.ascontiguousarray(local_values[self.send_local_ids], dtype=np.float32)
        recv = np.empty((int(self.recv_counts.sum()), nlev), dtype=np.float32)
        comm.Alltoallv(
            [send, ((self.send_counts * nlev).tolist(), _displacements(self.send_counts * nlev))],
            [recv, ((self.recv_counts * nlev).tolist(), _displacements(self.recv_counts * nlev))],
        )
        patch = np.empty((self.n_patch, nlev), dtype=np.float32)
        patch[self.recv_patch_pos] = recv
        return patch

    def from_patch(self, comm, patch_values: np.ndarray) -> np.ndarray:
        """Collective inverse of :meth:`to_patch`. ``patch_values``: (n_patch,
        nlev). Returns (n_owned, nlev) float32 for the cells
        ``send_local_ids``, in that order."""
        nlev = patch_values.shape[1]
        send = np.ascontiguousarray(patch_values[self.recv_patch_pos], dtype=np.float32)
        recv = np.empty((int(self.send_counts.sum()), nlev), dtype=np.float32)
        comm.Alltoallv(
            [send, ((self.recv_counts * nlev).tolist(), _displacements(self.recv_counts * nlev))],
            [recv, ((self.send_counts * nlev).tolist(), _displacements(self.send_counts * nlev))],
        )
        return recv
