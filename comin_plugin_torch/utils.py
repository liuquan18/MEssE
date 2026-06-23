import datetime
from pyparsing import Optional
import torch
import numpy as np
import cupy as xp
import healpy

import os
import sys
from mpi4py import MPI
import socket
import torch.distributed as dist
from dataclasses import dataclass

from yac import YAC, UnstructuredGrid, Location

from unet_online import UNetSnapshot, OnlineUNetTrainer

# ----------------------------------------------------------------------------
# MPI and PyTorch distributed setup
# ----------------------------------------------------------------------------


def setup_mpi_dist(comm):
    """PyTorch distributed initialization.
    ICON can launch more MPI tasks per node than physical GPUs (e.g. 5 tasks vs 4 GPUs).
    We use the SLURM local rank to identify which tasks are GPU-bearing.
    """
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    local_rank = int(os.environ.get("SLURM_LOCALID", "-1"))
    gpus_per_node = 4
    has_gpu = local_rank >= 0 and local_rank < gpus_per_node

    # Count how many MPI ranks are GPU-bearing globally.
    num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)

    print(
        f"[rank={rank}] MPI world size={world_size}, num_calculate_processes={num_calculate_processes}",
        file=sys.stderr,
    )

    print(
        f"[rank={rank}] local_rank={local_rank}, gpus_per_node={gpus_per_node}, "
        f"has_gpu={has_gpu}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        file=sys.stderr,
    )

    compute_comm: Optional[MPI.Comm] = None
    compute_rank: Optional[int] = None
    compute_size: Optional[int] = None

    if has_gpu:
        compute_comm = comm.Split(color=0, key=rank)
        compute_rank = compute_comm.Get_rank()
        compute_size = compute_comm.Get_size()

        master_addr = socket.gethostname() if compute_rank == 0 else None
        master_addr = compute_comm.bcast(master_addr, root=0)

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            rank=compute_rank,
            world_size=num_calculate_processes,
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    else:
        _ = comm.Split(color=1, key=rank)

    return compute_comm, compute_rank, compute_size, has_gpu


# ----------------------------------------------------------------------------
# YAC setup (HEALPix interpolation)
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


def parse_icon_datetime(iso_str: str) -> datetime.datetime:
    """Parse the ISO 8601 string returned by comin.current_get_datetime()."""
    clean = str(iso_str).split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc)

# Grid helpers
def extract_icon_cells(data_array, nc: int) -> xp.ndarray:
    """Return per-level data for this rank's owned ICON cells (halos excluded)."""
    data_xp = xp.asarray(data_array)
    # Drop trailing singleton dimensions added by COMIN
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]
    if data_xp.ndim == 2:
        return data_xp.ravel(order="F")[:nc]
    nlev = data_xp.shape[1]
    return data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]


def insert_icon_cells(pred_cells: np.ndarray, buffer) -> None:
    """Scatter (ncells, nlev) float64 array into COMIN (nproma, nlev, nblk) buffer in-place.

    Inverse of _extract_icon_cells: uses Fortran-order unraveling to match _extract_icon_cells.
    Fortran order: cell c in flattened array maps to buf[c % nproma, :, c // nproma].
    """
    nc, nlev = pred_cells.shape
    buf = xp.asarray(buffer)
    while buf.ndim > 3 and buf.shape[-1] == 1:
        buf = buf[..., 0]
    nproma_val = buf.shape[0]
    c = np.arange(nc)
    # Invert Fortran-order reshape: unravel index c in Fortran order
    buf[c % nproma_val, :, c // nproma_val] = xp.asarray(pred_cells).reshape(nc, nlev)


# ----------------------------------------------------------------------------
# Checkpoint load/save helpers
# ----------------------------------------------------------------------------


def save_checkpoint(
    trainer: OnlineUNetTrainer, checkpoint_path: str, compute_rank: int, step: int
) -> None:
    """Save model + optimizer state to CHECKPOINT_PATH.

    Only compute rank 0 writes to avoid concurrent writes on the shared filesystem.
    All other GPU ranks return immediately.
    """
    if compute_rank != 0:
        return
    # Write to a temporary file first, then rename for an atomic replace.
    tmp_path = checkpoint_path + ".tmp"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": step,
        },
        tmp_path,
    )
    os.replace(tmp_path, checkpoint_path)


def rollback_checkpoint(trainer: OnlineUNetTrainer, checkpoint_path) -> int:
    """Reload model + optimizer from the last saved checkpoint on all GPU ranks.

    Returns the step stored in the checkpoint, or 0 if no checkpoint exists.
    All compute ranks reload the same file so weights and optimizer moments stay
    consistent after a NaN-triggered recovery.
    """
    if not os.path.exists(checkpoint_path):
        return 0
    ckpt = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda", 0),
        weights_only=False,
    )
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def save_snapshot_pair(
    input_faces: torch.Tensor,
    pred_faces: torch.Tensor,
    step: int,
    compute_comm: MPI.Comm,
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


def sample_horizon(compute_rank, max_horizon) -> int:
    """Sample a random forecast horizon in [1, max_horizon].
    Only compute rank 0 samples the horizon, then broadcast to all ranks
    """
    horizon_t = torch.zeros(1, dtype=torch.int64, device="cuda")
    if compute_rank == 0:
        horizon_t[0] = torch.randint(1, max_horizon + 1, (1,)).item()
    dist.broadcast(horizon_t, src=0)
    return int(horizon_t.item())


# prepare data snapshot
@dataclass
class ForecastExample:
    source_snapshot: UNetSnapshot
    horizon: int
    due_step: int
    mean: torch.Tensor  # Per-level mean for denormalization
    std: torch.Tensor  # Per-level std for denormalization


def enqueue_snapshot(
    snapshot: UNetSnapshot,
    current_step: int,
    horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> ForecastExample:
    due_step = current_step + horizon

    return ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=due_step,
        mean=mean,
        std=std,
    )
