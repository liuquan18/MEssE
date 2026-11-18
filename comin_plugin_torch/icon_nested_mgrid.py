"""Nested ICON grids for online FieldSpaceNN training: one model patch per GPU
rank, built from complete R2B-backbone cells.

The MPI-M grid files (``/pool/data/ICON/grids/mpim/Earth_IcosS_*.nc``, R2B3 to
R2B9, same cells as ``public/mpim/00xx/icon_grid_00xx_R02Bnn_G.nc``) are
index-nested: the parent of cell ``c`` (0-based) is cell ``c // 4`` one level
up (checked from ``parent_cell_index`` on 2026-09-17). So the level-``base + k``
descendants of backbone cell ``b`` are the contiguous ids
``b * 4**k .. (b + 1) * 4**k - 1``, which is the nested ordering FieldSpaceNN's
``encode_zooms``/``decode_zooms`` and tokenizer expect.

Cell order
----------
In the file's order the corner children of a cell come in any order, so a
position inside a backbone cell is a different place in every one of them.
With ``canonical=True`` the grids come from FieldSpaceNN's
`icon_canonical_mgrid` instead: the same nesting, but every backbone cell has
its cells at the same places, and every cell has its 12 neighbours in the same
order. Each grid's ``file_index`` maps that order back to ICON's.

Patches
-------
ICON's domain decomposition ignores the hierarchy, so a backbone cell's fine
cells are usually spread over several ranks. Each backbone cell is therefore
given to the rank owning most of its fine cells, a rank's patch is the full
descendant tree of its backbone cells (:func:`patch_mgrid`), and
:class:`CellExchange` moves owned fine cells to the patch that holds them and
predictions back (``Alltoallv``). ICON's halo cells are not used.

A patch may also hold *ghost* backbone cells (:func:`ghost_cells`): cells of
other ranks that touch one of its own. They come after the owned ones, get
their values in the same exchange, and give the owned cells at the patch edge
their neighbours. Nothing is sent back for them.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from fieldspacenn.src.modules.grids.grid_utils import icon_grid_to_mgrid
from fieldspacenn.src.modules.grids.icon_grid import icon_canonical_mgrid

GRID_FILES = {
    3: "Earth_IcosS_0320km.nc",
    4: "Earth_IcosS_0160km.nc",
    5: "Earth_IcosS_0080km.nc",
    6: "Earth_IcosS_0040km.nc",
    7: "Earth_IcosS_0020km.nc",
    8: "Earth_IcosS_0010km.nc",
    9: "Earth_IcosS_0005km.nc",
}

# GridLayer's HEALPix polar fixup (run on adjacency columns 2 and 6) crashes
# on masked entries, so those columns are never masked in the file-order
# grids; the same neighbor is still masked in its other tiled columns.
_POLAR_FIX_COLUMNS = [2, 6]


def load_global_mgrid(grid_dir: str, base_level: int, fine_level: int, canonical: bool = False) -> List[Dict]:
    """FieldSpaceNN's ``mgrids`` for the whole globe, one entry per level from
    ``base_level`` to ``fine_level`` (coarsest first): in the files' cell
    order, or in canonical order (built from the finest file alone)."""
    if canonical:
        return icon_canonical_mgrid(os.path.join(grid_dir, GRID_FILES[fine_level]), fine_level - base_level + 1)
    return icon_grid_to_mgrid(
        {level: os.path.join(grid_dir, GRID_FILES[level]) for level in range(base_level, fine_level + 1)}
    )


def ghost_cells(owned: np.ndarray, touching: np.ndarray) -> np.ndarray:
    """The cells that touch one of ``owned`` without being one of them,
    sorted. ``touching``: (n_cells, k) every cell's neighbours, the cell
    itself where one is missing."""
    return np.setdiff1d(touching[owned], owned)


def patch_mgrid(global_mgrid: List[Dict], backbone_ids: torch.Tensor) -> List[Dict]:
    """``mgrids`` of the patch made of ``backbone_ids``, on every level, in
    their order. Cells are in nested order; neighbors outside the patch fall
    back to the cell itself and are masked."""
    patch = []
    for zoom, grid in enumerate(global_mgrid):
        ids = (backbone_ids[:, None] * 4**zoom + torch.arange(4**zoom)).flatten()
        row = torch.full((len(grid["adjc"]),), -1)  # patch row of every cell of the globe
        row[ids] = torch.arange(len(ids))
        local = row[grid["adjc"][ids]]
        inside = (local >= 0) & ~grid["adjc_mask"][ids]
        adjc_mask = ~inside
        if "grid_type" not in grid:
            adjc_mask[:, _POLAR_FIX_COLUMNS] = False
        cells = {key: value[ids] if torch.is_tensor(value) else value for key, value in grid.items()}
        cells.update(adjc=torch.where(inside, local, torch.arange(len(ids))[:, None]), adjc_mask=adjc_mask, zoom=zoom)
        patch.append(cells)
    return patch


def _offsets(counts: np.ndarray) -> List[int]:
    return [0, *np.cumsum(counts)[:-1].tolist()]


@dataclass
class CellExchange:
    """Routing of fine-cell values between each rank's owned ICON cells and
    the model patches. :meth:`to_patch` and :meth:`from_patch` are collective
    over the ranks of ``comm``."""

    send_ids: np.ndarray  # owned local cell ids, grouped by destination rank; once per patch that holds the cell
    send_counts: np.ndarray  # cells sent to each rank
    recv_counts: np.ndarray  # cells received from each rank
    recv_pos: np.ndarray  # patch row of every received cell
    home: np.ndarray  # of the sent cells: those going to the patch that owns them
    owned: np.ndarray  # of the received cells: those this patch owns (the others are ghosts)

    @classmethod
    def build(cls, comm, owned_local, owned_glb, owner, backbone_ids, refinement, touching: Optional[np.ndarray] = None):
        """``owned_local``/``owned_glb``: this rank's owned fine cells as local
        flat ids and 0-based global ids in the grids' cell order; ``owner``:
        rank of every backbone cell; ``backbone_ids``: this rank's patch;
        ``refinement``: fine cells per backbone cell. With ``touching``
        ((n_backbone, k) neighbours of every backbone cell, the cell itself
        where one is missing) a cell also goes to the patches that hold its
        backbone cell as a ghost."""
        n_ranks = comm.Get_size()
        backbone = owned_glb // refinement
        # A patch holds a backbone cell if its rank owns the cell or one that touches it.
        holders = owner[backbone][:, None]
        if touching is not None:
            holders = np.concatenate([holders, owner[touching[backbone]]], axis=1)
        pairs = np.unique(np.arange(len(backbone))[:, None] * n_ranks + holders)  # (cell, rank), each once
        cell, dest = pairs // n_ranks, pairs % n_ranks
        order = np.lexsort((owned_glb[cell], dest))
        cell, dest = cell[order], dest[order]

        send_counts = np.bincount(dest, minlength=n_ranks)
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_glb = np.empty(recv_counts.sum(), dtype=np.int64)
        comm.Alltoallv(
            [np.ascontiguousarray(owned_glb[cell]), (send_counts, _offsets(send_counts))],
            [recv_glb, (recv_counts, _offsets(recv_counts))],
        )
        row = np.full(len(owner), -1)  # patch row of every backbone cell
        row[backbone_ids] = np.arange(len(backbone_ids))
        recv_pos = row[recv_glb // refinement] * refinement + recv_glb % refinement
        return cls(
            owned_local[cell],
            send_counts,
            recv_counts,
            recv_pos,
            home=dest == owner[backbone[cell]],
            owned=owner[recv_glb // refinement] == comm.Get_rank(),
        )

    @property
    def owned_ids(self) -> np.ndarray:
        """Owned local cell ids in the order :meth:`from_patch` returns them."""
        return self.send_ids[self.home]

    def _alltoallv(self, comm, values, send_counts, recv_counts):
        nlev = values.shape[1]
        recv = np.empty((recv_counts.sum(), nlev), dtype=np.float32)
        comm.Alltoallv(
            [np.ascontiguousarray(values, dtype=np.float32), (send_counts * nlev, _offsets(send_counts * nlev))],
            [recv, (recv_counts * nlev, _offsets(recv_counts * nlev))],
        )
        return recv

    def to_patch(self, comm, local_values: np.ndarray) -> np.ndarray:
        """(n_local_cells, nlev) ICON cells of this rank -> (n_patch, nlev)
        patch in nested order, ghost cells included."""
        recv = self._alltoallv(comm, local_values[self.send_ids], self.send_counts, self.recv_counts)
        patch = np.empty_like(recv)
        patch[self.recv_pos] = recv
        return patch

    def from_patch(self, comm, patch_values: np.ndarray) -> np.ndarray:
        """The way back for the cells the patches own: (n_patch, nlev) ->
        values for the owned cells :attr:`owned_ids`, in that order."""
        # Cells per rank on the way back: each rank's share of the owned and of the home cells.
        from_ranks = np.repeat(np.arange(len(self.recv_counts)), self.recv_counts)[self.owned]
        to_ranks = np.repeat(np.arange(len(self.send_counts)), self.send_counts)[self.home]
        n_ranks = len(self.send_counts)
        return self._alltoallv(
            comm,
            patch_values[self.recv_pos[self.owned]],
            np.bincount(from_ranks, minlength=n_ranks),
            np.bincount(to_ranks, minlength=n_ranks),
        )
