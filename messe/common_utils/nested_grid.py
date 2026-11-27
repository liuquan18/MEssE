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
The grids therefore always come from FieldSpaceNN's `icon_canonical_mgrid`:
the same nesting, but every backbone cell has its cells at the same places,
and every cell has its 12 neighbours in the same order. Each grid's
``file_index`` maps that order back to ICON's.

Patches
-------
ICON's domain decomposition ignores the hierarchy, so a backbone cell's fine
cells are usually spread over several ranks. Each backbone cell is therefore
given to the rank owning most of its fine cells, a rank's patch is the full
descendant tree of its backbone cells (:func:`patch_mgrid`), and
:class:`CellExchange` moves owned fine cells to the patch that holds them and
predictions back (``Alltoallv``). ICON's halo cells are not used. The values
stay torch tensors on the GPU and go to MPI as they are: Levante's OpenMPI
(NVHPC, UCX) is CUDA-aware, so nothing is copied to the host.

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
# on masked entries, so those columns are never masked in grids without a
# ``grid_type`` (the tests' HEALPix-like grids); the same neighbor is still
# masked in its other tiled columns.
_POLAR_FIX_COLUMNS = [2, 6]


def load_global_mgrid(grid_dir: str, base_level: int, fine_level: int) -> List[Dict]:
    """FieldSpaceNN's ``mgrids`` for the whole globe, one entry per level from
    ``base_level`` to ``fine_level`` (coarsest first), in canonical order
    (built from the finest file alone)."""
    return icon_canonical_mgrid(os.path.join(grid_dir, GRID_FILES[fine_level]), fine_level - base_level + 1)


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
    over the ranks of ``comm`` and take and return float32 tensors on the
    exchange's device, which MPI reads and writes directly."""

    send_ids: torch.Tensor  # owned local cell ids, grouped by destination rank; once per patch that holds the cell
    send_counts: np.ndarray  # cells sent to each rank
    recv_counts: np.ndarray  # cells received from each rank
    recv_pos: torch.Tensor  # patch row of every received cell
    back_pos: torch.Tensor  # patch rows of the received cells this patch owns (not the ghosts): the way back
    back_counts: np.ndarray  # of those, cells going back to each rank
    home_counts: np.ndarray  # cells coming back from each rank: those this rank sent to their owning patch
    owned_ids: np.ndarray  # owned local cell ids in the order from_patch returns them (ICON's write-back index)

    @classmethod
    def build(cls, comm, owned_local, owned_glb, owner, backbone_ids, refinement, device, touching: Optional[np.ndarray] = None):
        """``owned_local``/``owned_glb``: this rank's owned fine cells as local
        flat ids and 0-based global ids in the grids' cell order; ``owner``:
        rank of every backbone cell; ``backbone_ids``: this rank's patch;
        ``refinement``: fine cells per backbone cell; ``device``: where the
        exchanged values live. With ``touching`` ((n_backbone, k) neighbours
        of every backbone cell, the cell itself where one is missing) a cell
        also goes to the patches that hold its backbone cell as a ghost."""
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
        owned = owner[recv_glb // refinement] == comm.Get_rank()  # of the received cells (the others are ghosts)
        home = dest == owner[backbone[cell]]  # of the sent cells: those going to the patch that owns them
        source = np.repeat(np.arange(n_ranks), recv_counts)  # rank of every received cell
        return cls(
            send_ids=torch.as_tensor(owned_local[cell], device=device),
            send_counts=send_counts,
            recv_counts=recv_counts,
            recv_pos=torch.as_tensor(recv_pos, device=device),
            back_pos=torch.as_tensor(recv_pos[owned], device=device),
            back_counts=np.bincount(source[owned], minlength=n_ranks),
            home_counts=np.bincount(dest[home], minlength=n_ranks),
            owned_ids=owned_local[cell][home],
        )

    @staticmethod
    def _alltoallv(comm, values: torch.Tensor, send_counts, recv_counts) -> torch.Tensor:
        """Rows of ``values`` to the ranks, ``send_counts`` each, as float32."""
        send = values.float().contiguous()
        nlev = send.shape[1]
        recv = send.new_empty((int(recv_counts.sum()), nlev))
        if send.is_cuda:  # MPI does not wait for the CUDA stream that wrote send
            torch.cuda.current_stream(send.device).synchronize()
        comm.Alltoallv(
            [send, (send_counts * nlev, _offsets(send_counts * nlev))],
            [recv, (recv_counts * nlev, _offsets(recv_counts * nlev))],
        )
        return recv

    def to_patch(self, comm, local_values: torch.Tensor) -> torch.Tensor:
        """(n_local_cells, nlev) ICON cells of this rank -> (n_patch, nlev)
        patch in nested order, ghost cells included."""
        recv = self._alltoallv(comm, local_values[self.send_ids], self.send_counts, self.recv_counts)
        patch = torch.empty_like(recv)
        patch[self.recv_pos] = recv
        return patch

    def from_patch(self, comm, patch_values: torch.Tensor) -> torch.Tensor:
        """The way back for the cells the patches own: (n_patch, nlev) ->
        values for the owned cells :attr:`owned_ids`, in that order."""
        return self._alltoallv(comm, patch_values[self.back_pos], self.back_counts, self.home_counts)
