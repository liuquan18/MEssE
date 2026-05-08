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

from mpi4py import MPI

from yac import (
    YAC,
    HealpixGrid,
    UnstructuredGrid,
    Location,
    Field,
    TimeUnit,
    InterpolationStack,
    NNNReductionType,
    Reduction
)

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
MAX_FORECAST_HORIZON = 30 # 30*120s = 1hour

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


# ----------------------------------------------------------------------------
# MPI setup
# ----------------------------------------------------------------------------
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()
world_size = comm.Get_size()

# ----------------------------------------------------------------------------
# PyTorch distributed initialization.
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
# comin / yac setup
# ----------------------------------------------------------------------------
glob = comin.descrdata_get_global()
domain = comin.descrdata_get_domain(DOMAIN_ID)

assert glob.yac_instance_id != -1, "The host-model is not configured with yac"
 
yac = YAC.from_id(glob.yac_instance_id)
source_comp = yac.predef_comp("icon_r2b4_source")

connectivity = (np.asarray(domain.cells.vertex_blk) - 1) * glob.nproma + (np.asarray(domain.cells.vertex_idx) - 1)

icon_grid = UnstructuredGrid(
    "icon_grid",
    np.ones(domain.cells.ncells, dtype=np.int32) * 3,
    np.array(np.ravel(np.transpose(domain.verts.vlon))[: domain.verts.nverts]),
    np.array(np.ravel(np.transpose(domain.verts.vlat))[: domain.verts.nverts]),
    np.ravel(np.swapaxes(connectivity, 0, 1))[: 3 * domain.cells.ncells]
)

icon_cell_centers = icon_grid.def_points(
    Location.CELL,
    np.ravel(domain.cells.clon)[: domain.cells.ncells],
    np.ravel(domain.cells.clat)[: domain.cells.ncells]
)


target_comp = yac.predef_comp("healpix_target")
# Define the HEALPix grid globally
hp_grid = HealpixGrid("hp_level6_grid", HPX_LEVEL)

# For parallel target: we distribute the 49152 HEALPix cells across GPU compute ranks only
total_hp_cells = 12 * 4**HPX_LEVEL
if has_gpu:
    cells_per_rank = total_hp_cells // num_calculate_processes
    start_idx = compute_rank * cells_per_rank
    # Last GPU rank takes the remainder
    end_idx = (compute_rank + 1) * cells_per_rank if compute_rank != num_calculate_processes - 1 else total_hp_cells
else:
    start_idx = 0
    end_idx = 0

# Each GPU rank defines a 'point set' covering only its local portion of the HEALPix grid.
# Non-GPU (IO) ranks define an empty point set to satisfy YAC coupling setup.
hp_points = hp_grid.def_points(Location.CELL, cell_indices=np.arange(start_idx, end_idx, dtype=np.uint64))



# ----------------------------------------------------------------------------
# Plugin state — all mutable runtime state lives here instead of as globals.
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        "current_step",
        "pending_example",
        "trainer",
        "icon_var",
        "hp_field_src",
        "hp_field_tgt",
        "step_len",
        "var_nlev",
        "owned_face_ids",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.pending_example: Optional["ForecastExample"] = None
        self.trainer: Optional[OnlineFieldSpaceNNTrainer] = None
        self.icon_var = None  # COMIN variable handle, set in sec_ctor
        self.hp_field_src: Optional[Field] = None  # YAC field on the ICON grid
        self.hp_field_tgt: Optional[Field] = None  # YAC field on the HEALPix grid
        self.step_len: Optional[float] = None  # Time
        self.var_nlev: Optional[int] = None  # Number of vertical levels in the ICON variable
        self.owned_face_ids: Optional[torch.Tensor] = None  # HEALPix face indices owned by this rank


_state = _State()

if has_gpu:
    _nside = 2**HPX_LEVEL
    _pixels_per_face = _nside * _nside
    _state.owned_face_ids = torch.arange(
        start_idx // _pixels_per_face,
        end_idx // _pixels_per_face,
        dtype=torch.long,
    )
else:
    _state.owned_face_ids = torch.tensor([], dtype=torch.long)


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


def _sample_horizon() -> int:
    """Sample a random forecast horizon in [1, MAX_FORECAST_HORIZON].

    The horizon is sampled on compute rank 0 and broadcast to all other
    compute ranks so that every GPU rank uses the same due_step — required
    for DDP ALLREDUCE to complete without deadlock.
    """
    horizon_t = torch.zeros(1, dtype=torch.int64, device="cuda")
    if compute_rank == 0:
        horizon_t[0] = torch.randint(1, MAX_FORECAST_HORIZON + 1, (1,)).item()
    dist.broadcast(horizon_t, src=0)
    return int(horizon_t.item())


def _enqueue_snapshot(
    snapshot: FieldSpaceNNSnapshot,
    current_step: int,
) -> ForecastExample:
    horizon = _sample_horizon()
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


@comin.register_callback(comin.EP_ATM_YAC_DEFCOMP_AFTER)
def setup_coupling():
    step_len_seconds = str(int(comin.descrdata_get_timesteplength(1)))
    _state.step_len = step_len_seconds
    # Derive nlev from the variable's memory layout: COMIN uses (nproma, nlev, nblk)
    # for 3-D fields and (nproma, nblk) for surface fields.
    _icon_arr = xp.asarray(_state.icon_var)
    while _icon_arr.ndim > 3 and _icon_arr.shape[-1] == 1:
        _icon_arr = _icon_arr[..., 0]
    _state.var_nlev = int(_icon_arr.shape[1]) if _icon_arr.ndim >= 3 else 1
    comin.print_info(f"[rank={rank}] timestep length: {_state.step_len} seconds")

    _state.hp_field_src = Field.create("var_remap", source_comp, icon_cell_centers, _state.var_nlev, _state.step_len, TimeUnit.SECOND)
    _state.hp_field_tgt = Field.create("var_target", target_comp, hp_points, _state.var_nlev, _state.step_len, TimeUnit.SECOND)

    interp = InterpolationStack()
    interp.add_nnn(NNNReductionType.AVG, n=1)

    yac.def_couple(
        "icon_r2b4_source", "icon_grid", "var_remap",
        "healpix_target", "hp_level6_grid", "var_target",
        _state.step_len, TimeUnit.SECOND, Reduction.TIME_NONE, interp
    )

@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online FieldSpaceNN training callback called by ICON at each time step."""
    # YAC put/get must be called on every MPI rank to avoid coupling deadlocks.
    # source: ICON native grid → target: HEALPix grid
    icon_cells = _extract_icon_cells(_state.icon_var)  # (ncells,) or (ncells, nlev)
    _state.hp_field_src.put(icon_cells, inplace=True)
    var_hpx, info = _state.hp_field_tgt.get()

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

    if var_hpx is None or len(var_hpx) == 0:
        comin.print_info(
            f"[rank={rank}] HPX regrid not ready, skipping step {current_step}"
        )
        return

    # Convert numpy/cupy array from YAC to a float32 GPU tensor of shape (n_pixels, nlev).
    var_hpx_t = torch.as_tensor(xp.asarray(var_hpx), device="cuda").float()
    if var_hpx_t.ndim == 1:
        var_hpx_t = var_hpx_t.unsqueeze(-1)

    ua_faces = _to_hpx_faces(var_hpx_t)
    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=_state.var_nlev)
    snapshot = trainer.prepare_snapshot(ua_faces, unix_seconds)

    if _state.pending_example is None:
        # First effective timestep: store source snapshot and wait for target.
        _state.pending_example = _enqueue_snapshot(snapshot, current_step)
    else:
        # Due step: train on (source, target) pair, then re-enqueue current as source.
        result = trainer.train_step(_state.pending_example.source_snapshot, snapshot)
        comin.print_info(
            f"[rank={rank}] step={current_step} loss={result['loss']:.6f} "
            f"horizon={_state.pending_example.horizon}"
        )
        _state.pending_example = _enqueue_snapshot(snapshot, current_step)
