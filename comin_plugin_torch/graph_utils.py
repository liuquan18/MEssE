"""Build a local, patch-based graph directly on the ICON native grid.

Design summary
--------------
Each MPI/GPU rank in ICON owns a *domain-decomposed patch* of the global
triangular mesh: a contiguous set of "owned" (prognostic) cells plus a ring
of "halo" cells that are mirrored from neighboring ranks so that ICON's
finite-volume/finite-difference stencils can be evaluated locally.

We reuse exactly this decomposition as the GNN's patch/sample boundary:

* Graph nodes  = all local cells returned by COMIN for this rank
  (owned cells *and* halo cells), addressed with the same flat index
  ``c = (block - 1) * nproma + (index - 1)`` used by
  :func:`utils.extract_icon_cells` / :func:`utils.insert_icon_cells`.
* Graph edges  = the native ICON cell-to-cell adjacency (each triangular
  cell has up to 3 neighbors), taken from
  ``domain.cells.neighbor_idx`` / ``domain.cells.neighbor_blk``.
  Edges are added in both directions so message passing is symmetric.
* Halo cells are *included* as graph nodes so that message passing for
  owned cells near the patch boundary sees correct neighbor information
  (otherwise those boundary nodes would be missing up to 2 of their 3
  neighbors). The loss/metrics are only ever evaluated on *owned* nodes
  (``owned_mask``), so halo cells never contribute directly to the loss;
  they only pass messages.

This avoids YAC and HEALPix regridding entirely: no interpolation, and the
domain decomposition (and therefore the GPU layout / DDP world) is
inherited unmodified from the ICON model.

Open design question (see README): halo cells are refreshed by ICON's own
MPI halo exchange before COMIN hands us the field, so their values are
always in sync with the owning rank at the current timestep. We do not
run an extra halo exchange for the GNN itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class LocalGraph:
    """Static (topology never changes during a run) local patch graph."""

    n_nodes: int  # nproma * nblks_c for this rank (includes unused padding)
    n_valid: int  # domain.cells.ncells: local cells actually in use
    edge_index: torch.Tensor  # (2, E) int64, local node ids, both directions
    owned_mask: torch.Tensor  # (n_nodes,) bool; True for prognostic (non-halo) cells
    clon: torch.Tensor  # (n_nodes,) float32, radians
    clat: torch.Tensor  # (n_nodes,) float32, radians
    edge_attr: torch.Tensor  # (E, 3) float32: [dlon, dlat, great_circle_dist]


def _flat_index(idx: np.ndarray, blk: np.ndarray, nproma: int) -> np.ndarray:
    """Convert 1-based (idx, blk) COMIN indices to a 0-based flat cell id.

    Matches the flattening convention used by
    ``utils.extract_icon_cells``/``utils.insert_icon_cells`` and by
    ``utils.setup_icon_grid`` when building the YAC connectivity, i.e.
    ``c = (blk - 1) * nproma + (idx - 1)``.
    """
    return (blk.astype(np.int64) - 1) * nproma + (idx.astype(np.int64) - 1)


def _great_circle(lon1, lat1, lon2, lat2):
    """Vectorized haversine great-circle distance (radians in, radians out)."""
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_local_graph(
    domain,
    nproma: int,
    device: Optional[torch.device] = None,
    add_self_loops: bool = True,
) -> LocalGraph:
    """Build the local patch graph for this rank from COMIN descriptive data.

    Parameters
    ----------
    domain:
        Return value of ``comin.descrdata_get_domain(DOMAIN_ID)``.
    nproma:
        ``glob.nproma`` (block length), needed to flatten (idx, blk) pairs.
    device:
        Where to place the resulting tensors (defaults to CPU; move to GPU
        once, since the topology is static for the whole run).
    add_self_loops:
        Whether to add a self-loop per node (common in GNN message passing
        so a node's own state directly informs its update).
    """
    n_valid = int(domain.cells.ncells)
    n_nodes = int(domain.cells.nblks) * nproma

    clon_full = np.ravel(domain.cells.clon).astype(np.float64)
    clat_full = np.ravel(domain.cells.clat).astype(np.float64)
    # decomp_domain == 0 marks prognostic ("owned") cells; > 0 marks halo
    # cells at increasing distance from the owned region; < 0 is unused
    # padding at the end of the last block.
    decomp = np.ravel(domain.cells.decomp_domain).astype(np.int64)
    owned = decomp == 0

    neighbor_idx = np.asarray(domain.cells.neighbor_idx)  # (nproma, nblks_c, 3)
    neighbor_blk = np.asarray(domain.cells.neighbor_blk)
    n_nbr = neighbor_idx.shape[-1]

    # Build "self" flat ids aligned with (nproma, nblks_c, n_nbr) layout.
    nproma_ids, blk_ids = np.meshgrid(
        np.arange(1, nproma + 1), np.arange(1, neighbor_idx.shape[1] + 1), indexing="ij"
    )
    self_flat = _flat_index(nproma_ids, blk_ids, nproma)  # (nproma, nblks_c)
    self_flat = np.repeat(self_flat[:, :, None], n_nbr, axis=2)  # (nproma, nblks_c, 3)

    nbr_flat = _flat_index(neighbor_idx, neighbor_blk, nproma)

    src = self_flat.ravel()
    dst = nbr_flat.ravel()

    # A missing neighbor (open lateral boundary of a limited-area nest, or
    # unused padding cell) is encoded by ICON with idx/blk <= 0.
    valid = (
        (neighbor_idx.ravel() > 0)
        & (neighbor_blk.ravel() > 0)
        & (src >= 0)
        & (src < n_nodes)
        & (dst >= 0)
        & (dst < n_nodes)
    )
    src = src[valid]
    dst = dst[valid]

    # Message passing both ways (the raw connectivity is already symmetric
    # for interior cells, but this also fixes any asymmetry from halo cells
    # whose own neighbor list was not shipped to this rank).
    src_bidir = np.concatenate([src, dst])
    dst_bidir = np.concatenate([dst, src])

    if add_self_loops:
        self_ids = np.arange(n_nodes)
        src_bidir = np.concatenate([src_bidir, self_ids])
        dst_bidir = np.concatenate([dst_bidir, self_ids])

    # Drop duplicate edges (bidirectional + self-loop construction can
    # create repeats near halo boundaries).
    edges = np.unique(np.stack([src_bidir, dst_bidir], axis=0), axis=1)
    src_bidir, dst_bidir = edges[0], edges[1]

    edge_dist = _great_circle(
        clon_full[src_bidir], clat_full[src_bidir], clon_full[dst_bidir], clat_full[dst_bidir]
    )
    edge_dlon = clon_full[dst_bidir] - clon_full[src_bidir]
    edge_dlat = clat_full[dst_bidir] - clat_full[src_bidir]
    edge_attr = np.stack([edge_dlon, edge_dlat, edge_dist], axis=-1)

    edge_index_t = torch.as_tensor(edges, dtype=torch.int64)
    owned_t = torch.as_tensor(owned, dtype=torch.bool)
    clon_t = torch.as_tensor(clon_full, dtype=torch.float32)
    clat_t = torch.as_tensor(clat_full, dtype=torch.float32)
    edge_attr_t = torch.as_tensor(edge_attr, dtype=torch.float32)

    if device is not None:
        edge_index_t = edge_index_t.to(device)
        owned_t = owned_t.to(device)
        clon_t = clon_t.to(device)
        clat_t = clat_t.to(device)
        edge_attr_t = edge_attr_t.to(device)

    return LocalGraph(
        n_nodes=n_nodes,
        n_valid=n_valid,
        edge_index=edge_index_t,
        owned_mask=owned_t,
        clon=clon_t,
        clat=clat_t,
        edge_attr=edge_attr_t,
    )
