"""Moving data between ICON (through ComIn) and PyTorch, used by every plugin.

- `setup_mpi_dist`: split ICON's MPI ranks into GPU ranks and start DDP;
- `extract_icon_cells` / `insert_icon_cells` / `flat_index`: ComIn's (nproma, nlev, nblks)
  arrays <-> flat per-cell arrays;
- `parse_icon_datetime`, `sample_interval`: ICON's clock.

``cupy`` is optional: without a usable GPU, array ops fall back to numpy. ``mpi4py`` is
imported inside `setup_mpi_dist` only, because importing it calls ``MPI_Init``, which
aborts the process outside an MPI job (e.g. in the unit tests).
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

try:
    import cupy as xp

    xp.cuda.runtime.getDeviceCount()  # raises if cupy has no usable CUDA device
except Exception:
    # Falls back to numpy both when cupy isn't importable (e.g. an
    # installed-but-incompatible build, wrong numpy ABI) *and* when it
    # imports fine but there is no usable GPU/driver here (e.g. a login
    # node) — `import cupy` succeeding does not mean a device is actually
    # usable, and any array op (even a plain asarray) raises in that case.
    # Either way, numpy is the correct fallback: everything this module
    # returns is immediately wrapped in a torch tensor, so the array
    # backend choice here is purely a GPU-avoids-a-host-copy optimization,
    # not a correctness requirement.
    xp = np


# ----------------------------------------------------------------------------
# MPI and PyTorch distributed setup
# ----------------------------------------------------------------------------


def setup_mpi_dist(comm):
    """PyTorch distributed initialization.
    ICON can launch more MPI tasks per node than physical GPUs (e.g. 5 tasks vs 4 GPUs).
    We use the SLURM local rank to identify which tasks are GPU-bearing.

    ``mpi4py`` is imported lazily here (not at module level) because merely
    importing it triggers an implicit ``MPI_Init`` — fine inside an actual
    ICON/MPI job, but it aborts the process outside one (e.g. a plain
    interactive shell on a login node). Keeping the import local means the
    rest of this module (the pure COMIN<->tensor helpers) stays importable
    — and unit-testable — without an MPI job context; only this function,
    which genuinely needs a live MPI communicator, pays that cost.
    """
    from mpi4py import MPI

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
# COMIN <-> tensor data marshalling
# ----------------------------------------------------------------------------


def flat_index(idx: np.ndarray, blk: np.ndarray, nproma: int) -> np.ndarray:
    """Convert 1-based (idx, blk) COMIN indices to a 0-based flat cell id.

    ``c = (blk - 1) * nproma + (idx - 1)``. This is the flattening convention
    used throughout this module (:func:`extract_icon_cells`/
    :func:`insert_icon_cells`) and by every local-patch grid/graph builder
    that addresses cells in COMIN's native (nproma, nblks) layout
    (``graph.build_local_graph``, ``two_level_mgrid``).
    """
    return (blk.astype(np.int64) - 1) * nproma + (idx.astype(np.int64) - 1)


def sample_interval(wanted_seconds: float, step_seconds: float) -> Tuple[int, int]:
    """Turn a wanted sampling interval in model *seconds* into whole ICON
    steps, given the run's timestep (``comin.descrdata_get_timesteplength``).

    A plugin that acts every n-th step should be configured in seconds, so the
    same setting keeps its meaning when the timestep changes with resolution
    (e.g. 60 s at R2B8 today, 10 s later). Returns ``(stride_steps,
    interval_seconds)`` with ``interval_seconds = stride_steps *
    step_seconds``: the interval actually realizable, which differs from
    ``wanted_seconds`` when that is not a multiple of the timestep. The stride
    is at least one step.
    """
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds}")
    stride = max(1, int(round(float(wanted_seconds) / float(step_seconds))))
    return stride, int(round(stride * float(step_seconds)))


def parse_icon_datetime(iso_str: str) -> datetime.datetime:
    """Parse the ISO 8601 string returned by comin.current_get_datetime()."""
    clean = str(iso_str).split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc)


def extract_icon_cells(data_array, nc: int):
    """Return per-level data for this rank's local ICON cells (owned + halo).

    ``nc`` (typically ``domain.cells.ncells``) is the number of cells this
    rank actually holds data for, i.e. owned ("prognostic") cells *and* the
    halo ring mirrored from neighboring ranks; it does **not** exclude
    halos. Truncating to ``nc`` only drops the unused padding cells beyond
    ``nc`` (up to ``nproma * nblks``). Callers that must not use/emit halo
    values (e.g. final predictions written back to ICON) need to further
    filter by ``domain.cells.decomp_domain == 0`` (see
    :func:`insert_icon_cells`'s ``indices`` argument and
    ``graph.LocalGraph.owned_mask``).
    """
    data_xp = xp.asarray(data_array)
    # Drop trailing singleton dimensions added by COMIN
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]
    if data_xp.ndim == 2:
        return data_xp.ravel(order="F")[:nc]
    nlev = data_xp.shape[1]
    return data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]


def insert_icon_cells(pred_cells: np.ndarray, buffer, indices: Optional[np.ndarray] = None) -> None:
    """Scatter a (n, nlev) float64 array into COMIN (nproma, nlev, nblk) buffer in-place.

    Inverse of extract_icon_cells: uses Fortran-order unraveling to match extract_icon_cells.
    Fortran order: cell c in flattened array maps to buf[c % nproma, :, c // nproma].

    Parameters
    ----------
    pred_cells:
        (n, nlev) array of values to scatter.
    buffer:
        COMIN (nproma, nlev, nblk)-shaped buffer to write into.
    indices:
        Optional flat cell ids (same 0-based Fortran-order convention as
        extract_icon_cells) that each row of ``pred_cells`` should be
        written to. Defaults to ``np.arange(n)``, i.e. the first ``n``
        cells (owned + halo, no reordering). Pass e.g. the owned-cell ids
        from ``graph.LocalGraph.owned_mask`` to only write
        predictions for owned cells and leave halo cells untouched.
    """
    n, nlev = pred_cells.shape
    buf = xp.asarray(buffer)
    while buf.ndim > 3 and buf.shape[-1] == 1:
        buf = buf[..., 0]
    nproma_val = buf.shape[0]
    c = np.arange(n) if indices is None else np.asarray(indices)
    # Invert Fortran-order reshape: unravel index c in Fortran order
    buf[c % nproma_val, :, c // nproma_val] = xp.asarray(pred_cells).reshape(n, nlev)
