"""Generic, YAC/HEALPix-independent helpers for online (in-simulation)
training: moving data between COMIN and PyTorch, MPI/DDP process-group
bootstrap, checkpoint I/O, online normalization statistics, and turning a
stream of per-timestep snapshots into training examples.

Every online-training plugin (`unet_plugin.py`, `gnn_plugin.py`, and the
eventual FieldSpaceNN plugin) needs all of this, and none of it is specific
to a model architecture or to a grid (HEALPix vs. ICON native) — nothing
here assumes an interpolation step (YAC) or a specific target grid. This is
what lets plugins that train directly on the ICON native grid
(`gnn_plugin.py`) depend on this module without dragging in `yac`/`healpy`
at import time. The YAC- and HEALPix-specific counterparts (grid setup,
HEALPix face reshaping) used by the interpolation-based UNet plugin live in
`healpix_grids.py`, a sibling module in this package.

Two schemes are used for turning per-timestep pushes into a (source,
target) training example, both single-target (one loss evaluation per
example, not one per intermediate step) but differing in how the horizon
is chosen:
  - `RolloutBuffer` — a *fixed* horizon of ``n_steps``: waits for
    ``n_steps`` pushes after a source, then pairs the source with that
    single target. The model still rolls forward autoregressively through
    the ``n_steps`` gap using its own predictions (see
    `gnn_online.OnlineGNNTrainer.train_rollout_step`) — only the loss
    evaluation is single-target, not the forward computation.
    (`gnn_plugin.py`.)
  - `sample_horizon` / `ForecastExample` / `enqueue_snapshot` — a
    *randomly sampled* horizon each round instead of a fixed one,
    otherwise the same single-target idea. (`unet_plugin.py`.)

``cupy`` is optional: if a CUDA-capable cupy is not importable (no GPU, or
a numpy/cupy version mismatch), array ops fall back to numpy. Every array
this module hands back is immediately wrapped in a torch tensor by
callers, so correctness only depends on producing a plain array-like
object. `mpi4py` is imported lazily inside `setup_mpi_dist` (not at module
level): merely importing it triggers an implicit `MPI_Init`, which aborts
the process outside a real MPI job on some builds — keeping the import
local means the rest of this module stays importable, and unit-testable,
without an MPI job context.
"""

from __future__ import annotations

import datetime
import os
import socket
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
# Online normalization statistics
# ----------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor


class RunningMeanStd:
    """Accumulates per-node mean/std over a fixed number of samples, then freezes.

    Pure tensor bookkeeping — the caller is responsible for feeding it one
    already-extracted sample per step via :meth:`update`.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        n_samples: int,
        device: torch.device,
        eps: float = 1e-6,
    ) -> None:
        self.n_samples = int(n_samples)
        self.eps = eps
        self.count = 0
        self._sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self._sumsq = torch.zeros(shape, dtype=torch.float64, device=device)
        self.stats: Optional[NormalizationStats] = None

    @property
    def done(self) -> bool:
        return self.stats is not None

    def update(self, x: torch.Tensor) -> bool:
        """Accumulate one sample. Returns True once statistics are frozen.

        No-op (returns True immediately) once already done, so callers can
        call this unconditionally every step without checking ``done`` first.
        """
        if self.done:
            return True
        x64 = x.double()
        self._sum += x64
        self._sumsq += x64 * x64
        self.count += 1
        if self.count >= self.n_samples:
            mean = (self._sum / self.count).float()
            e_x2 = (self._sumsq / self.count).float()
            var = torch.clamp(e_x2 - mean * mean, min=self.eps)
            self.stats = NormalizationStats(mean=mean, std=torch.sqrt(var))
        return self.done

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.stats is not None, "RunningMeanStd: statistics not finalized yet"
        return (x - self.stats.mean) / self.stats.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert self.stats is not None, "RunningMeanStd: statistics not finalized yet"
        return x * self.stats.std + self.stats.mean


# ----------------------------------------------------------------------------
# Fixed-horizon, single-target scheduling
# ----------------------------------------------------------------------------


class RolloutBuffer:
    """Accumulates a source snapshot and waits ``n_steps`` pushes for a
    single target ``n_steps`` steps later.

    ``n_steps`` is a horizon *distance*, not a window length: the loss is
    evaluated exactly once per training example, against the one real
    snapshot observed ``n_steps`` steps after the source — snapshots pushed
    in between are not kept or compared against. The model computation
    itself still rolls forward autoregressively through that ``n_steps``
    gap, feeding each step's own prediction back in (see
    `gnn_online.OnlineGNNTrainer.train_rollout_step`); this class only
    controls *when* the loss gets evaluated, not how many forward passes
    the model takes to get there. For a horizon that varies randomly each
    round instead of a fixed one, see :func:`sample_horizon` below.

    Pure bookkeeping over whatever snapshot object is pushed in — no
    `comin`/MPI dependency — so it is unit-testable standalone.
    """

    def __init__(self, n_steps: int) -> None:
        self.n_steps = int(n_steps)
        self.source: Optional[Any] = None
        self._pushes_since_source = 0

    def push(self, snapshot: Any) -> Optional[Tuple[Any, Any]]:
        """Push the latest snapshot.

        Returns ``(source, target)`` once ``n_steps`` pushes have occurred
        since the current source was set, else ``None``. After a pair is
        returned, the just-pushed snapshot becomes the *next* example's
        source — the horizon slides forward rather than resetting to empty.
        """
        if self.source is None:
            self.source = snapshot
            self._pushes_since_source = 0
            return None
        self._pushes_since_source += 1
        if self._pushes_since_source < self.n_steps:
            return None
        ready = (self.source, snapshot)
        self.source = snapshot
        self._pushes_since_source = 0
        return ready

    def reset(self) -> None:
        self.source = None
        self._pushes_since_source = 0


# ----------------------------------------------------------------------------
# Single-target, variable-horizon scheduling
# ----------------------------------------------------------------------------


def sample_horizon(compute_rank: int, max_horizon: int) -> int:
    """Sample a random forecast horizon in [1, max_horizon].
    Only compute rank 0 samples the horizon, then broadcasts it to all ranks
    so every rank trains on the same horizon this round.
    """
    horizon_t = torch.zeros(1, dtype=torch.int64, device="cuda")
    if compute_rank == 0:
        horizon_t[0] = torch.randint(1, max_horizon + 1, (1,)).item()
    dist.broadcast(horizon_t, src=0)
    return int(horizon_t.item())


@dataclass
class ForecastExample:
    """A pending (source, due future step) training example: source ->
    a single target ``horizon`` steps ahead. ``mean``/``std`` ride along so
    the prediction can be denormalized once the target snapshot arrives,
    keeping the example self-contained regardless of the source snapshot
    type (works with any model's snapshot dataclass, e.g.
    `unet_online.UNetSnapshot`)."""

    source_snapshot: Any
    horizon: int
    due_step: int
    mean: torch.Tensor  # Per-level mean for denormalization
    std: torch.Tensor  # Per-level std for denormalization


def enqueue_snapshot(
    snapshot: Any,
    current_step: int,
    horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> ForecastExample:
    return ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=current_step + horizon,
        mean=mean,
        std=std,
    )


# ----------------------------------------------------------------------------
# COMIN <-> tensor data marshalling
# ----------------------------------------------------------------------------


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
    ``graph_utils.LocalGraph.owned_mask``).
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
        from ``graph_utils.LocalGraph.owned_mask`` to only write
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


# ----------------------------------------------------------------------------
# Checkpoint load/save helpers
# ----------------------------------------------------------------------------


def save_checkpoint(trainer: Any, checkpoint_path: str, compute_rank: int, step: int) -> None:
    """Save model + optimizer state to ``checkpoint_path``.

    ``trainer`` is any object exposing ``.model`` and ``.optimizer``
    (``gnn_online.OnlineGNNTrainer``, ``unet_online.OnlineUNetTrainer``, ...).
    Only compute rank 0 writes to avoid concurrent writes on the shared
    filesystem; all other GPU ranks return immediately.
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


def rollback_checkpoint(trainer: Any, checkpoint_path: str) -> int:
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
    return int(ckpt.get("step", 0))
