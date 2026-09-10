"""YAC + HEALPix-specific grid setup and face-reshaping helpers.

Used by the interpolation-based UNet plugin (`unet_plugin.py`), which
regrids the ICON native grid to HEALPix via YAC before training. Kept
separate from `icon_online_helper.py` (the generic COMIN<->tensor helpers
and online-training bookkeeping) on purpose: plugins that train directly on
the ICON native grid (`gnn_plugin.py`, and the eventual native-grid
FieldSpaceNN plugin — see CLAUDE.md) have no business depending on YAC or
HEALPix at all. Native-grid training is the direction this project is
headed in, so this module is expected to shrink, not grow.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

import healpy
from yac import YAC, UnstructuredGrid, Location


# ----------------------------------------------------------------------------
# YAC grid setup
# ----------------------------------------------------------------------------


def setup_icon_grid(yac, glob, domain, has_gpu):
    """Define the ICON grid in YAC and get the cell center points from comin"""
    if has_gpu:
        source_comp = yac.predef_comp("icon_r2b4_source")

        connectivity = (np.asarray(domain.cells.vertex_blk) - 1) * glob.nproma + (
            np.asarray(domain.cells.vertex_idx) - 1
        )

        icon_grid = UnstructuredGrid(
            "icon_grid",
            np.ones(domain.cells.ncells, dtype=np.int32) * 3,
            np.array(np.ravel(np.transpose(domain.verts.vlon))[: domain.verts.nverts]),
            np.array(np.ravel(np.transpose(domain.verts.vlat))[: domain.verts.nverts]),
            np.ravel(np.swapaxes(connectivity, 0, 1))[: 3 * domain.cells.ncells],
        )

        icon_cell_centers = icon_grid.def_points(
            Location.CELL,
            np.ravel(domain.cells.clon)[: domain.cells.ncells],
            np.ravel(domain.cells.clat)[: domain.cells.ncells],
        )

    else:
        source_comp = None

        icon_grid = None
        icon_cell_centers = None

    return source_comp, icon_grid, icon_cell_centers


# Build HEALPix target grid with healpy
def _xyz2lonlat(xyz):
    xyz = np.array(xyz)
    lat = np.arcsin(xyz[..., 2])
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    return lon, lat


def _make_healpix_grid(name, nside, nest=True, cell_idx=None):
    if cell_idx is None:
        ncells = healpy.pixelfunc.nside2npix(nside)
        cell_idx = np.arange(ncells)

    centers_xyz = np.stack(
        healpy.pixelfunc.pix2vec(nside, cell_idx, nest=nest),
        axis=-1,
    )
    clon, clat = _xyz2lonlat(centers_xyz)

    boundaries_xyz = (
        healpy.boundaries(nside, cell_idx, nest=nest).transpose(0, 2, 1).reshape(-1, 3)
    )
    verts_xyz, quads = np.unique(boundaries_xyz, return_inverse=True, axis=0)
    vlon, vlat = _xyz2lonlat(verts_xyz)
    vertex_of_cell = quads.reshape(-1, 4)

    grid = UnstructuredGrid(
        name,
        np.full(len(cell_idx), 4, dtype=np.int32),
        vlon,
        vlat,
        vertex_of_cell.flatten(),
    )
    points = grid.def_points(Location.CELL, clon, clat)
    return grid, points


def _local_hp_indices(HPX_LEVEL, compute_rank, compute_size):
    nside = 2**HPX_LEVEL
    pixels_per_face = nside * nside
    total_faces = 12
    faces_per_rank = total_faces // compute_size
    extra_faces = total_faces % compute_size

    start_face = compute_rank * faces_per_rank + min(compute_rank, extra_faces)
    n_local_faces = faces_per_rank + (1 if compute_rank < extra_faces else 0)
    end_face = start_face + n_local_faces

    start_idx = start_face * pixels_per_face
    end_idx = end_face * pixels_per_face

    return np.arange(start_idx, end_idx)


def setup_hpx_grid(yac, HPX_LEVEL, rank, num_proc, has_gpu):
    """Define the HEALPix target grid in YAC.
    If has_gpu is False, return None for all outputs.
    """

    if has_gpu:
        local_hp_indices = _local_hp_indices(HPX_LEVEL, rank, num_proc)
        target_comp = yac.predef_comp("healpix_target")
        hpx_grid, hpx_points = _make_healpix_grid(
            f"hp_level{HPX_LEVEL}_grid", nside=2**HPX_LEVEL, cell_idx=local_hp_indices
        )

    else:
        target_comp = None
        hpx_grid = None
        hpx_points = None

    return target_comp, hpx_grid, hpx_points


# ----------------------------------------------------------------------------
# HEALPix face <-> flat pixel reshaping
# ----------------------------------------------------------------------------


def to_hpx_faces(owned_vals: torch.Tensor, hpx_level) -> torch.Tensor:
    """Reshape (n_owned_pixels, nlev) to (faces_per_rank, nlev, nside, nside)."""
    nside = 2**hpx_level
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)

    return (
        owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()
    )


def from_hpx_faces(pred_faces: torch.Tensor) -> torch.Tensor:
    """Reshape (faces_per_rank, nlev, nside, nside) back to (n_owned_pixels, nlev).

    Inverse of to_hpx_faces.
    """
    return pred_faces.permute(0, 2, 3, 1).contiguous().reshape(-1, pred_faces.shape[1])


# ----------------------------------------------------------------------------
# HEALPix-face snapshot saving
# ----------------------------------------------------------------------------


def save_snapshot_pair(
    input_faces: torch.Tensor,
    pred_faces: torch.Tensor,
    step: int,
    # Typed loosely (not `mpi4py.MPI.Comm`) to avoid importing mpi4py at
    # module level: a bare `from mpi4py import MPI` triggers an implicit
    # MPI_Init that aborts the process outside a real MPI job on this
    # build (see icon_io.py's setup_mpi_dist docstring). Any object with
    # an mpi4py-Comm-shaped `.gather()` works.
    compute_comm: Any,
    compute_rank: int,
    SNAPSHOTS_DIR: str,
) -> None:
    """Gather HEALPix face pairs from all compute ranks and save to disk (rank 0 only).

    Saves a compressed .npz file with:
      input : (12, nlev, nside, nside)  — denormalized ICON field on HEALPix faces
      pred  : (12, nlev, nside, nside)  — UNet prediction on HEALPix faces
      step  : scalar int
    Faces are stored in NESTED HEALPix ordering, matching the plugin convention.
    """
    local_inp = input_faces.cpu().numpy()  # (local_faces, nlev, nside, nside)
    local_pred = pred_faces.cpu().numpy()
    all_inp = compute_comm.gather(local_inp, root=0)
    all_pred = compute_comm.gather(local_pred, root=0)
    if compute_rank != 0:
        return
    full_inp = np.concatenate(all_inp, axis=0)  # (12, nlev, nside, nside)
    full_pred = np.concatenate(all_pred, axis=0)
    snap_path = os.path.join(SNAPSHOTS_DIR, f"snapshot_step{step:06d}.npz")
    np.savez_compressed(snap_path, input=full_inp, pred=full_pred, step=step)
