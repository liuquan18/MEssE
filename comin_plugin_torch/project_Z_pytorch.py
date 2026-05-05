import comin
import datetime
import os
import socket
import sys
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
    # Fall back to the directory provided by the environment or cwd.
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
# GPU check
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
    comin.print_info("No NVIDIA device found falling back to numpy")
    import numpy as xp

    DEVICE_SYNC_FLAG = 0


domain = comin.descrdata_get_domain(DOMAIN_ID)

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()
world_size = comm.Get_size()

# ----------------------------------------------------------------------------
# PyTorch distributed initialization.
# The run wrapper gives GPU-bearing ranks exactly one CUDA_VISIBLE_DEVICES entry.
# Non-GPU ranks never enter NCCL or the online FieldSpaceNN trainer.
# ----------------------------------------------------------------------------
_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
has_gpu = _cuda_vis.isdigit()

num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)
num_no_gpu_processes = world_size - num_calculate_processes

compute_rank = None

if has_gpu:
    compute_comm = comm.Split(color=0, key=rank)
    compute_rank = compute_comm.Get_rank()

    if compute_rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None
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


_current_step = 0
_pending_example = None
_online_trainer = None

# ---------------------------------------------------------------------------
# HEALPix regridding globals, initialized once for GPU ranks.
# ---------------------------------------------------------------------------
_hpx_regridder = None
_n_owned_pixels = 0
_faces_per_rank = 0
_owned_face_ids = None


class ForecastExample:
    def __init__(
        self,
        source_snapshot: FieldSpaceNNSnapshot,
        horizon: int,
        due_step: int,
        source_unix_seconds: float,
    ) -> None:
        self.source_snapshot = source_snapshot
        self.horizon = horizon
        self.due_step = due_step
        self.source_unix_seconds = source_unix_seconds


def _parse_icon_datetime(iso_str: str) -> datetime.datetime:
    """Parse the ISO 8601 string returned by comin.current_get_datetime()."""
    clean = str(iso_str).split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc)


def _icon_time_unix_seconds() -> float:
    return float(_parse_icon_datetime(comin.current_get_datetime()).timestamp())


def _init_hpx_regridding():
    """Build the per-rank ICON-to-owned-HEALPix nearest-neighbour regridder."""
    global _hpx_regridder, _n_owned_pixels, _faces_per_rank, _owned_face_ids

    nc = domain.cells.ncells
    n_compute = compute_comm.Get_size()

    if 12 % n_compute != 0:
        comin.print_info(
            f"[rank={rank}] WARNING: n_compute={n_compute} does not divide 12; "
            "face-based HEALPix ownership is uneven, regrid disabled"
        )
        return

    nside = 2**HPX_LEVEL
    faces_per_rank = 12 // n_compute
    first_face = compute_rank * faces_per_rank
    last_face = first_face + faces_per_rank

    owned_pixel_ids = np.arange(
        first_face * nside**2, last_face * nside**2, dtype=np.int64
    )
    _n_owned_pixels = len(owned_pixel_ids)
    _faces_per_rank = faces_per_rank
    _owned_face_ids = torch.arange(
        first_face, last_face, device="cuda", dtype=torch.long
    )

    comin.print_info(
        f"[rank={rank}/cr={compute_rank}] HPX init: faces {first_face}-{last_face - 1}, "
        f"{_n_owned_pixels} pixels"
    )

    grid = e2g_healpix.Grid(level=HPX_LEVEL, pixel_order=e2g_healpix.PixelOrder.NEST)
    owned_ids_t = torch.tensor(owned_pixel_ids, dtype=torch.long)
    owned_lon, owned_lat = grid.pix2ang(owned_ids_t)

    cell_lon = np.rad2deg(
        np.asarray(domain.cells.clon, dtype=np.float64).ravel(order="F")
    )[:nc]
    cell_lat = np.rad2deg(
        np.asarray(domain.cells.clat, dtype=np.float64).ravel(order="F")
    )[:nc]

    src_lon_t = torch.tensor(cell_lon, dtype=torch.float32)
    src_lat_t = torch.tensor(cell_lat, dtype=torch.float32)

    regridder = KNNS2Interpolator(
        src_lon=src_lon_t.flatten(),
        src_lat=src_lat_t.flatten(),
        dest_lon=owned_lon.float().flatten(),
        dest_lat=owned_lat.float().flatten(),
        k=1,
    )
    _hpx_regridder = regridder.cuda()
    comin.print_info(
        f"[rank={rank}] HPX regridder ready: {nc} src cells -> {_n_owned_pixels} owned pixels"
    )


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    global icon_var

    icon_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (ICON_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )

    if has_gpu:
        _init_hpx_regridding()


def all_local_data(data_array):
    """Extract all local cells, including halo cells, preserving level dimension."""
    nc = domain.cells.ncells
    data_xp = xp.asarray(data_array)
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]
    if data_xp.ndim == 2:
        return data_xp.ravel(order="F")[:nc]
    nlev = data_xp.shape[1]
    return data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]


def regrid_to_healpix(data_array) -> Optional[torch.Tensor]:
    """Regrid ICON cells to this rank's owned HEALPix pixels."""
    if _hpx_regridder is None or _n_owned_pixels == 0:
        return None
    cells = all_local_data(data_array)
    cells_t = tdlpack.from_dlpack(cells).float()
    owned = _hpx_regridder(cells_t.T)
    return owned.T.contiguous()


def to_hpx_faces(owned_vals: torch.Tensor) -> torch.Tensor:
    """Reshape (n_owned, nlev) to (faces_per_rank, nlev, nside, nside)."""
    nside = 2**HPX_LEVEL
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)
    return (
        owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()
    )


def _get_online_trainer(nlev: int) -> OnlineFieldSpaceNNTrainer:
    global _online_trainer

    if _online_trainer is not None:
        if int(_online_trainer.nlev) != int(nlev):
            raise RuntimeError(
                f"ICON level count changed from {_online_trainer.nlev} to {nlev}"
            )
        return _online_trainer

    if _owned_face_ids is None:
        raise RuntimeError("HEALPix face ownership is not initialized.")

    _online_trainer = OnlineFieldSpaceNNTrainer(
        cfg=_ONLINE_CFG,
        owned_face_ids=_owned_face_ids,
        nlev=int(nlev),
        device=torch.device("cuda", 0),
        use_ddp=dist.is_initialized(),
        log_fn=comin.print_info,
        rank=rank,
    )
    comin.print_info(
        f"[rank={rank}] FieldSpaceNN MG_Transformer initialised: "
        f"zooms={_online_trainer.in_zooms}, horizon={_online_trainer.forecast_horizon_steps}"
    )
    return _online_trainer


def _enqueue_snapshot(
    snapshot: FieldSpaceNNSnapshot, current_step: int
) -> ForecastExample:
    horizon = int(_online_trainer.forecast_horizon_steps)
    example = ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=current_step + horizon,
        source_unix_seconds=snapshot.unix_seconds,
    )
    comin.print_info(
        f"[rank={rank}] enqueued FieldSpaceNN source at step={current_step}, "
        f"horizon={horizon}, due_step={example.due_step}"
    )
    return example


def process_step(
    mode: str, snapshot: Optional[FieldSpaceNNSnapshot], current_step: int
):
    """Run the online FieldSpaceNN state machine for one COMIN callback."""
    global _pending_example

    if mode == "waiting":
        comin.print_info(
            f"[rank={rank}] step={current_step}, waiting "
            f"(due at step={_pending_example.due_step}, horizon={_pending_example.horizon})"
        )
        return

    if snapshot is None:
        raise RuntimeError("FieldSpaceNN snapshot is required outside waiting mode.")

    if mode == "initial":
        _pending_example = _enqueue_snapshot(snapshot, current_step)
        return

    if mode == "training":
        result = _online_trainer.train_step(_pending_example.source_snapshot, snapshot)
        comin.print_info(
            f"[rank={rank}] trained FieldSpaceNN step={current_step}, "
            f"horizon={_pending_example.horizon}, loss={result['loss']:.6f}"
        )
        _pending_example = _enqueue_snapshot(snapshot, current_step)
        return

    raise ValueError(f"Unknown process_step mode: {mode}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online FieldSpaceNN training callback called by ICON at each time step."""
    if not has_gpu:
        return

    global _current_step

    current_step = _current_step
    _current_step += 1

    if _pending_example is not None and current_step < _pending_example.due_step:
        process_step(mode="waiting", snapshot=None, current_step=current_step)
        return

    ua_hpx = regrid_to_healpix(icon_var)
    if ua_hpx is None:
        comin.print_info(
            f"[rank={rank}] HPX regrid not ready, skipping step {current_step}"
        )
        return

    comin.print_info(
        f"[rank={rank}] step={current_step} HPX output: "
        f"shape={ua_hpx.shape}, dtype={ua_hpx.dtype}"
    )
    ua_local = to_hpx_faces(ua_hpx)
    comin.print_info(f"[rank={rank}] HPX faces: shape={ua_local.shape}")

    unix_seconds = _icon_time_unix_seconds()

    if _pending_example is None:
        # First effective timestep: initialize the model once with random weights.
        trainer = _get_online_trainer(nlev=ua_local.shape[1])
        snapshot = trainer.prepare_snapshot(ua_local, unix_seconds=unix_seconds)
        process_step(mode="initial", snapshot=snapshot, current_step=current_step)
    else:
        # Subsequent training steps: reuse the already-initialized model.
        snapshot = _online_trainer.prepare_snapshot(ua_local, unix_seconds=unix_seconds)
        process_step(mode="training", snapshot=snapshot, current_step=current_step)
