import comin
import datetime
import os
import socket
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.dlpack as tdlpack

from mpi4py import MPI
import earth2grid.healpix as e2g_healpix
from earth2grid._regrid import KNNS2Interpolator

try:
    _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may be undefined when COMIN loads the plugin via exec().
    _PLUGIN_DIR = os.environ.get("MESSE_PLUGIN_DIR", os.getcwd())
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from fieldspacenn_online import (
    FieldSpaceNNSnapshot,
    OnlineFieldSpaceNNTrainer,
    load_online_config,
)


_ONLINE_CFG = load_online_config()
DOMAIN_ID = int(_ONLINE_CFG.online.variable.domain_id)
ICON_VARIABLE_NAME = str(_ONLINE_CFG.online.variable.icon_name)
HPX_LEVEL = int(_ONLINE_CFG.online.hpx_level)

# ----------------------------------------------------------------------------
# GPU / array backend selection
# ----------------------------------------------------------------------------
glob = comin.descrdata_get_global()
if glob.has_device:
    comin.print_info(f"glob.device_name={glob.device_name}")
    comin.print_info(f"glob.device_vendor={glob.device_vendor}")
    comin.print_info(f"glob.device_driver={glob.device_driver}")

if glob.has_device and "NVIDIA" in glob.device_vendor.upper():
    try:
        comin.print_info("Using cupy!")
        import cupy as xp

        DEVICE_SYNC_FLAG = comin.COMIN_FLAG_DEVICE
    except ImportError as e:
        comin.print_info("Cannot import cupy, falling back to numpy")
        comin.print_info(e)
        comin.print_info(sys.path)
        import numpy as xp

        DEVICE_SYNC_FLAG = 0
else:
    comin.print_info("No NVIDIA device found, falling back to numpy")
    import numpy as xp

    DEVICE_SYNC_FLAG = 0

domain = comin.descrdata_get_domain(DOMAIN_ID)

# ----------------------------------------------------------------------------
# MPI setup
# ----------------------------------------------------------------------------
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()
world_size = comm.Get_size()

# ----------------------------------------------------------------------------
# PyTorch distributed initialization.
# The run wrapper gives GPU-bearing ranks exactly one CUDA_VISIBLE_DEVICES entry.
# Non-GPU ranks never enter NCCL or the online FieldSpaceNN trainer.
# ----------------------------------------------------------------------------
_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# A single digit means exactly one GPU was assigned to this rank.
has_gpu = _cuda_vis.isdigit()

num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)

compute_rank: Optional[int] = None
compute_comm: Optional[MPI.Comm] = None

if has_gpu:
    compute_comm = comm.Split(color=0, key=rank)
    compute_rank = compute_comm.Get_rank()

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
    comin.print_info(
        f"[rank={rank}] PyTorch distributed initialised: "
        f"compute_rank={compute_rank}/{num_calculate_processes}"
    )
else:
    _ = comm.Split(color=1, key=rank)


# ----------------------------------------------------------------------------
# Plugin state — all mutable runtime state lives here instead of as globals.
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        "current_step",
        "pending_example",
        "trainer",
        "regridder",
        "n_owned_pixels",
        "faces_per_rank",
        "owned_face_ids",
        "icon_var",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.pending_example: Optional["ForecastExample"] = None
        self.trainer: Optional[OnlineFieldSpaceNNTrainer] = None
        self.regridder = None  # KNNS2Interpolator, set after init
        self.n_owned_pixels: int = 0
        self.faces_per_rank: int = 0
        self.owned_face_ids: Optional[torch.Tensor] = None
        self.icon_var = None  # COMIN variable handle, set in sec_ctor


_state = _State()


@dataclass
class ForecastExample:
    source_snapshot: FieldSpaceNNSnapshot
    horizon: int
    due_step: int
    source_unix_seconds: float


# ----------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------

def _parse_icon_datetime(iso_str: str) -> datetime.datetime:
    """Parse the ISO 8601 string returned by comin.current_get_datetime()."""
    clean = str(iso_str).split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc)


def _icon_time_unix_seconds() -> float:
    return float(_parse_icon_datetime(comin.current_get_datetime()).timestamp())


# ----------------------------------------------------------------------------
# HEALPix regridding
# ----------------------------------------------------------------------------

def _init_hpx_regridding(compute_comm: MPI.Comm, compute_rank: int) -> None:
    """Build the per-rank ICON-to-owned-HEALPix nearest-neighbour regridder."""
    n_compute = compute_comm.Get_size()
    if 12 % n_compute != 0:
        comin.print_info(
            f"[rank={rank}] WARNING: n_compute={n_compute} does not evenly divide "
            "12 HEALPix faces; regridding disabled."
        )
        return

    nside = 2**HPX_LEVEL
    faces_per_rank = 12 // n_compute
    first_face = compute_rank * faces_per_rank

    owned_pixel_ids = np.arange(
        first_face * nside**2,
        (first_face + faces_per_rank) * nside**2,
        dtype=np.int64,
    )

    grid = e2g_healpix.Grid(level=HPX_LEVEL, pixel_order=e2g_healpix.PixelOrder.NEST)
    owned_lon, owned_lat = grid.pix2ang(torch.tensor(owned_pixel_ids, dtype=torch.long))

    nc = domain.cells.ncells
    cell_lon = np.rad2deg(
        np.asarray(domain.cells.clon, dtype=np.float64).ravel(order="F")[:nc]
    )
    cell_lat = np.rad2deg(
        np.asarray(domain.cells.clat, dtype=np.float64).ravel(order="F")[:nc]
    )

    regridder = KNNS2Interpolator(
        src_lon=torch.tensor(cell_lon, dtype=torch.float32),
        src_lat=torch.tensor(cell_lat, dtype=torch.float32),
        dest_lon=owned_lon.float(),
        dest_lat=owned_lat.float(),
        k=1,
    ).cuda()

    _state.n_owned_pixels = len(owned_pixel_ids)
    _state.faces_per_rank = faces_per_rank
    _state.owned_face_ids = torch.arange(
        first_face, first_face + faces_per_rank, device="cuda", dtype=torch.long
    )
    _state.regridder = regridder

    comin.print_info(
        f"[rank={rank}/cr={compute_rank}] HPX: faces {first_face}–"
        f"{first_face + faces_per_rank - 1}, {_state.n_owned_pixels} pixels, "
        f"{nc} src cells"
    )


def _extract_icon_cells(data_array) -> xp.ndarray:
    """Return per-level data for this rank's owned ICON cells (halos excluded)."""
    nc = domain.cells.ncells
    data_xp = xp.asarray(data_array)
    # Drop trailing singleton dimensions added by COMIN
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]
    if data_xp.ndim == 2:
        return data_xp.ravel(order="F")[:nc]
    nlev = data_xp.shape[1]
    return data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]


@torch.no_grad()
def _regrid_to_healpix(data_array) -> Optional[torch.Tensor]:
    """Regrid ICON cells to this rank's owned HEALPix pixels."""
    if _state.regridder is None or _state.n_owned_pixels == 0:
        return None
    cells = _extract_icon_cells(data_array)
    cells_t = tdlpack.from_dlpack(cells).float()
    return _state.regridder(cells_t.T).T.contiguous()


def _to_hpx_faces(owned_vals: torch.Tensor) -> torch.Tensor:
    """Reshape (n_owned_pixels, nlev) to (faces_per_rank, nlev, nside, nside)."""
    nside = 2**HPX_LEVEL
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)
    return owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------
# Trainer lifecycle
# ----------------------------------------------------------------------------

def _get_trainer(nlev: int) -> OnlineFieldSpaceNNTrainer:
    if _state.trainer is not None:
        if _state.trainer.nlev != nlev:
            raise RuntimeError(
                f"ICON level count changed: {_state.trainer.nlev} → {nlev}"
            )
        return _state.trainer

    if _state.owned_face_ids is None:
        raise RuntimeError("HEALPix face ownership is not initialized.")

    _state.trainer = OnlineFieldSpaceNNTrainer(
        cfg=_ONLINE_CFG,
        owned_face_ids=_state.owned_face_ids,
        nlev=nlev,
        device=torch.device("cuda", 0),
        use_ddp=dist.is_initialized(),
        log_fn=comin.print_info,
        rank=rank,
    )
    comin.print_info(
        f"[rank={rank}] FieldSpaceNN trainer initialized: "
        f"zooms={_state.trainer.in_zooms}, "
        f"horizon={_state.trainer.forecast_horizon_steps}"
    )
    return _state.trainer


def _enqueue_snapshot(
    snapshot: FieldSpaceNNSnapshot,
    current_step: int,
    trainer: OnlineFieldSpaceNNTrainer,
) -> ForecastExample:
    horizon = trainer.forecast_horizon_steps
    due_step = current_step + horizon
    comin.print_info(
        f"[rank={rank}] enqueued snapshot at step={current_step}, due={due_step}"
    )
    return ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=due_step,
        source_unix_seconds=snapshot.unix_seconds,
    )


# ----------------------------------------------------------------------------
# COMIN callbacks
# ----------------------------------------------------------------------------

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    _state.icon_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (ICON_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )
    if has_gpu:
        _init_hpx_regridding(compute_comm, compute_rank)


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online FieldSpaceNN training callback called by ICON at each time step."""
    if not has_gpu:
        return

    current_step = _state.current_step
    _state.current_step += 1

    # Not yet due — wait for the target snapshot
    if _state.pending_example is not None and current_step < _state.pending_example.due_step:
        comin.print_info(
            f"[rank={rank}] step={current_step} waiting "
            f"(due={_state.pending_example.due_step})"
        )
        return

    ua_hpx = _regrid_to_healpix(_state.icon_var)
    if ua_hpx is None:
        comin.print_info(
            f"[rank={rank}] HPX regrid not ready, skipping step {current_step}"
        )
        return

    ua_faces = _to_hpx_faces(ua_hpx)
    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=ua_faces.shape[1])
    snapshot = trainer.prepare_snapshot(ua_faces, unix_seconds)

    if _state.pending_example is None:
        # First effective timestep: store source snapshot and wait for target.
        _state.pending_example = _enqueue_snapshot(snapshot, current_step, trainer)
    else:
        # Due step: train on (source, target) pair, then re-enqueue current as source.
        result = trainer.train_step(_state.pending_example.source_snapshot, snapshot)
        comin.print_info(
            f"[rank={rank}] step={current_step} loss={result['loss']:.6f} "
            f"horizon={_state.pending_example.horizon}"
        )
        _state.pending_example = _enqueue_snapshot(snapshot, current_step, trainer)
