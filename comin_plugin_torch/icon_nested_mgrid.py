"""Nested ICON grids for online FieldSpaceNN training: one model patch per GPU
rank, built from complete R2B-backbone cells.

The MPI-M grid files (``/pool/data/ICON/grids/mpim/Earth_IcosS_*.nc``, R2B3 to
R2B9, same cells as ``public/mpim/00xx/icon_grid_00xx_R02Bnn_G.nc``) are
index-nested: the parent of cell ``c`` (0-based) is cell ``c // 4`` one level
up (checked from ``parent_cell_index`` on 2026-09-17). So the level-``base + k``
descendants of backbone cell ``b`` are the contiguous ids
``b * 4**k .. (b + 1) * 4**k - 1``, which is the nested ordering FieldSpaceNN's
``encode_zooms``/``decode_zooms`` and tokenizer expect.

ICON's domain decomposition ignores this hierarchy, so a backbone cell's fine
cells are usually spread over several ranks. Each backbone cell is therefore
given to the rank owning most of its fine cells, a rank's patch is the full
descendant tree of its backbone cells (:func:`patch_mgrid`), and
:class:`CellExchange` moves owned fine cells to the patch that holds them and
predictions back (``Alltoallv``). Every fine cell is in exactly one patch.
Halo cells are not used.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from fieldspacenn.src.modules.grids.grid_utils import icon_grid_to_mgrid

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
# on masked entries, so those columns are never masked; the same neighbor is
# still masked in its other tiled columns.
_POLAR_FIX_COLUMNS = [2, 6]


def load_global_mgrid(grid_dir: str, base_level: int, fine_level: int) -> List[Dict]:
    """FieldSpaceNN's ``mgrids`` for the whole globe, one entry per level from
    ``base_level`` to ``fine_level`` (coarsest first)."""
    return icon_grid_to_mgrid(
        {level: os.path.join(grid_dir, GRID_FILES[level]) for level in range(base_level, fine_level + 1)}
    )


def patch_mgrid(global_mgrid: List[Dict], backbone_ids: torch.Tensor) -> List[Dict]:
    """``mgrids`` of the patch made of the (sorted) ``backbone_ids``, on every
    level. Cells are in nested order; neighbors outside the patch fall back to
    the cell itself and are masked."""
    patch = []
    for zoom, grid in enumerate(global_mgrid):
        ids = (backbone_ids[:, None] * 4**zoom + torch.arange(4**zoom)).flatten()
        neighbors = grid["adjc"][ids]
        local = torch.searchsorted(ids, neighbors).clamp(max=len(ids) - 1)
        inside = ids[local] == neighbors
        adjc_mask = ~inside
        adjc_mask[:, _POLAR_FIX_COLUMNS] = False
        patch.append(
            {
                "coords": grid["coords"][ids],
                "adjc": torch.where(inside, local, torch.arange(len(ids))[:, None]),
                "adjc_mask": adjc_mask,
                "zoom": zoom,
            }
        )
    return patch


def _offsets(counts: np.ndarray) -> List[int]:
    return [0, *np.cumsum(counts)[:-1].tolist()]


@dataclass
class CellExchange:
    """Routing of fine-cell values between each rank's owned ICON cells and
    the model patches. :meth:`to_patch` and :meth:`from_patch` are collective
    over the ranks of ``comm``."""

    send_ids: np.ndarray  # owned local cell ids, grouped by destination rank
    send_counts: np.ndarray  # cells sent to each rank
    recv_counts: np.ndarray  # cells received from each rank
    recv_pos: np.ndarray  # patch row of every received cell

    @classmethod
    def build(cls, comm, owned_local, owned_glb, owner, backbone_ids, refinement):
        """``owned_local``/``owned_glb``: this rank's owned fine cells as local
        flat ids and 0-based global ids; ``owner``: rank of every backbone
        cell; ``backbone_ids``: this rank's (sorted) backbone cells;
        ``refinement``: fine cells per backbone cell."""
        dest = owner[owned_glb // refinement]
        order = np.lexsort((owned_glb, dest))
        send_counts = np.bincount(dest, minlength=comm.Get_size())
        recv_counts = np.empty_like(send_counts)
        comm.Alltoall(send_counts, recv_counts)
        recv_glb = np.empty(recv_counts.sum(), dtype=np.int64)
        comm.Alltoallv(
            [np.ascontiguousarray(owned_glb[order]), (send_counts, _offsets(send_counts))],
            [recv_glb, (recv_counts, _offsets(recv_counts))],
        )
        recv_pos = np.searchsorted(backbone_ids, recv_glb // refinement) * refinement + recv_glb % refinement
        return cls(owned_local[order], send_counts, recv_counts, recv_pos)

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
        patch in nested order."""
        recv = self._alltoallv(comm, local_values[self.send_ids], self.send_counts, self.recv_counts)
        patch = np.empty_like(recv)
        patch[self.recv_pos] = recv
        return patch

    def from_patch(self, comm, patch_values: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_patch`: (n_patch, nlev) -> values for the
        owned cells ``send_ids``, in that order."""
        return self._alltoallv(comm, patch_values[self.recv_pos], self.recv_counts, self.send_counts)
