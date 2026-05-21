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

import healpy
from yac import (
    YAC,
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
FORCING_VARIABLE_NAME = str(_ONLINE_CFG.online.variable.forcing_name)
HPX_LEVEL = int(_ONLINE_CFG.online.hpx_level)
MAX_FORECAST_HORIZON = int(_ONLINE_CFG.online.forecast_horizon_maxsteps)
EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "fieldspacenn_online.pt")
SAVE_INTERVAL_SECONDS: int = 86400  # P1D — save the model the same frequency as icon output

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
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    comin.print_info(
        f"[rank={rank}] PyTorch distributed initialised: "
        f"compute_rank={compute_rank}/{num_calculate_processes}"
    )
else:
    _ = comm.Split(color=1, key=rank)



# ----------------------------------------------------------------------------
# yac setup (HEALPix interpolation)
# ----------------------------------------------------------------------------
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

# For parallel target: distribute HEALPix cells across GPU compute ranks only
total_hp_cells = 12 * 4**HPX_LEVEL
if has_gpu:
    cells_per_rank = total_hp_cells // num_calculate_processes
    start_idx = compute_rank * cells_per_rank
    # Last GPU rank takes the remainder
    end_idx = (compute_rank + 1) * cells_per_rank if compute_rank != num_calculate_processes - 1 else total_hp_cells
else:
    start_idx = 0
    end_idx = 0

# Build HEALPix target grid with healpy 
nside = 2**HPX_LEVEL
local_hp_indices = np.arange(start_idx, end_idx)


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
        healpy.boundaries(nside, cell_idx, nest=nest)
        .transpose(0, 2, 1)
        .reshape(-1, 3)
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


if len(local_hp_indices) > 0:
    hp_grid, hp_points = _make_healpix_grid(
        "hp_level6_grid", nside, cell_idx=local_hp_indices
    )
else:
    # IO / non-compute rank: empty partition to satisfy YAC coupling setup
    hp_grid = UnstructuredGrid(
        "hp_level6_grid",
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0, dtype=np.int32),
    )
    hp_points = hp_grid.def_points(Location.CELL, np.zeros(0), np.zeros(0))


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
# Plugin state — all mutable runtime state lives here instead of as globals.
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        "current_step",
        "pending_example",
        "trainer",
        "icon_var",
        "icon_forcing",
        "hp_field_src",
        "hp_field_tgt",
        "hp_forcing_src",
        "hp_forcing_tgt",
        "var_nlev",
        "owned_face_ids",
        "step_len_seconds",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.pending_example: Optional["ForecastExample"] = None
        self.trainer: Optional[OnlineFieldSpaceNNTrainer] = None
        self.icon_var = None       # COMIN variable handle for ua, set in sec_ctor
        self.icon_forcing = None   # COMIN variable handle for ts forcing, set in sec_ctor
        self.hp_field_src: Optional[Field] = None   # YAC field for ua on ICON grid
        self.hp_field_tgt: Optional[Field] = None   # YAC field for ua on HEALPix grid
        self.hp_forcing_src: Optional[Field] = None  # YAC field for forcing on ICON grid
        self.hp_forcing_tgt: Optional[Field] = None  # YAC field for forcing on HEALPix grid
        self.var_nlev: Optional[int] = None  # Combined nlev (ua + forcing), set in setup_coupling
        self.owned_face_ids: Optional[torch.Tensor] = None  # HEALPix face indices owned by this rank
        self.step_len_seconds: Optional[int] = None  # ICON timestep length in seconds


_state = _State()

if has_gpu:
    _pixels_per_face = nside * nside
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
    )
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location=_state.trainer.device,
            weights_only=False,
        )
        _state.trainer.model.load_state_dict(ckpt["model_state_dict"])
        _state.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        comin.print_info(
            f"[rank={rank}] Loaded checkpoint from {CHECKPOINT_PATH} "
            f"(step={ckpt.get('step', 'unknown')})"
        )
    return _state.trainer


def _sample_horizon() -> int:
    """Sample a random forecast horizon in [1, MAX_FORECAST_HORIZON].
    Only compute rank 0 samples the horizon, then broadcast to all ranks
    """
    horizon_t = torch.zeros(1, dtype=torch.int64, device="cuda")
    if compute_rank == 0:
        horizon_t[0] = torch.randint(1, MAX_FORECAST_HORIZON + 1, (1,)).item()
    dist.broadcast(horizon_t, src=0)
    return int(horizon_t.item())


def _enqueue_snapshot(
    snapshot: FieldSpaceNNSnapshot,
    current_step: int,
    horizon: int,
) -> ForecastExample:
    due_step = current_step + horizon
    comin.print_info(
        f"[rank={rank}] enqueued snapshot at step={current_step}, due={due_step}"
    )
    return ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=due_step,
    )


# ----------------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------------


def _save_checkpoint(trainer: OnlineFieldSpaceNNTrainer, step: int) -> None:
    """Save model + optimizer state to CHECKPOINT_PATH.

    Only compute rank 0 writes to avoid concurrent writes on the shared filesystem.
    All other GPU ranks return immediately.
    """
    if compute_rank != 0:
        return
    # Write to a temporary file first, then rename for an atomic replace.
    tmp_path = CHECKPOINT_PATH + ".tmp"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "step": step,
        },
        tmp_path,
    )
    os.replace(tmp_path, CHECKPOINT_PATH)
    comin.print_info(
        f"[rank={rank}] Checkpoint saved at step={step} → {CHECKPOINT_PATH}"
    )


# ----------------------------------------------------------------------------
# COMIN callbacks
# ----------------------------------------------------------------------------

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    # the variable
    _state.icon_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (ICON_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )

    # forcing (sst, but use ts here)
    _state.icon_forcing = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (FORCING_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )


@comin.register_callback(comin.EP_ATM_YAC_DEFCOMP_AFTER)
def setup_coupling():

    step_len = str(int(comin.descrdata_get_timesteplength(1)))
    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    # Derive nlev from each variable's memory layout: COMIN uses (nproma, nlev, nblk)
    # for 3-D fields and (nproma, nblk) for surface fields.

    _icon_arr = xp.asarray(_state.icon_var)
    while _icon_arr.ndim > 3 and _icon_arr.shape[-1] == 1:
        _icon_arr = _icon_arr[..., 0]
    var_nlev = int(_icon_arr.shape[1]) if _icon_arr.ndim >= 3 else 1

    _forcing_arr = xp.asarray(_state.icon_forcing)
    while _forcing_arr.ndim > 3 and _forcing_arr.shape[-1] == 1:
        _forcing_arr = _forcing_arr[..., 0]
    forcing_nlev = int(_forcing_arr.shape[1]) if _forcing_arr.ndim >= 3 else 1

    # Store combined depth for trainer initialization
    _state.var_nlev = var_nlev + forcing_nlev

    comin.print_info(f"[rank={rank}] timestep length: {step_len} seconds")
    comin.print_info(f"[rank={rank}] ua nlev={var_nlev}, forcing nlev={forcing_nlev}, combined={_state.var_nlev}")

    _state.hp_field_src = Field.create("var_remap", source_comp, icon_cell_centers, var_nlev, step_len, TimeUnit.SECOND)
    _state.hp_field_tgt = Field.create("var_target", target_comp, hp_points, var_nlev, step_len, TimeUnit.SECOND)

    _state.hp_forcing_src = Field.create("forcing_remap", source_comp, icon_cell_centers, forcing_nlev, step_len, TimeUnit.SECOND)
    _state.hp_forcing_tgt = Field.create("forcing_target", target_comp, hp_points, forcing_nlev, step_len, TimeUnit.SECOND)

    interp = InterpolationStack()
    interp.add_nnn(NNNReductionType.AVG, n=1)

    yac.def_couple(
        "icon_r2b4_source", "icon_grid", "var_remap",
        "healpix_target", "hp_level6_grid", "var_target",
        step_len, TimeUnit.SECOND, Reduction.TIME_NONE, interp
    )
    yac.def_couple(
        "icon_r2b4_source", "icon_grid", "forcing_remap",
        "healpix_target", "hp_level6_grid", "forcing_target",
        step_len, TimeUnit.SECOND, Reduction.TIME_NONE, interp
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online FieldSpaceNN training callback called by ICON at each time step."""

    if not has_gpu:
        return

    current_step = _state.current_step
    _state.current_step += 1

    # Periodic checkpoint save — independent of training/waiting state.
    if (
        _state.trainer is not None
        and _state.step_len_seconds is not None
        and current_step > 0
    ):
        steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
        if current_step % steps_per_save == 0:
            _save_checkpoint(_state.trainer, current_step)

    # Mode: Waiting; between t and t+horizon
    if _state.pending_example is not None and current_step < _state.pending_example.due_step:
        comin.print_info(
            f"[rank={rank}] step={current_step} waiting "
            f"horizon={_state.pending_example.horizon} "
            f"(due={_state.pending_example.due_step})"
        )
        return

    icon_var_cells = _extract_icon_cells(_state.icon_var)       # (ncells, nlev_ua)
    icon_forcing_cells = _extract_icon_cells(_state.icon_forcing)  # (ncells,) or (ncells, 1)

    # Interpolation from native ICON grid to HEALPix using YAC
    # currently YAC only supports numpy arrays, wait for update to support cupy arrays
    icon_var_cells_np = icon_var_cells.get() if hasattr(icon_var_cells, 'get') else icon_var_cells
    icon_forcing_cells_np = icon_forcing_cells.get() if hasattr(icon_forcing_cells, 'get') else icon_forcing_cells
    _state.hp_field_src.put(icon_var_cells_np)
    _state.hp_forcing_src.put(icon_forcing_cells_np)
    var_hpx, info = _state.hp_field_tgt.get()
    forcing_hpx, _ = _state.hp_forcing_tgt.get()

    # Convert numpy/cupy arrays from YAC to float32 GPU tensors of shape (n_pixels, nlev).
    # YAC returns (nlev, ncells) for multi-level fields and (ncells,) for single-level.
    var_hpx_t = torch.as_tensor(xp.asarray(var_hpx), device="cuda").float()
    if var_hpx_t.ndim == 1:
        var_hpx_t = var_hpx_t.unsqueeze(-1)
    elif var_hpx_t.ndim == 2:
        var_hpx_t = var_hpx_t.T  # (nlev, ncells) -> (ncells, nlev)

    forcing_hpx_t = torch.as_tensor(xp.asarray(forcing_hpx), device="cuda").float()
    if forcing_hpx_t.ndim == 1:
        forcing_hpx_t = forcing_hpx_t.unsqueeze(-1)
    elif forcing_hpx_t.ndim == 2:
        forcing_hpx_t = forcing_hpx_t.T  # (nlev, ncells) -> (ncells, nlev)

    # Concatenate ua and forcing along the level dimension: (ncells, nlev_ua + forcing_nlev)
    combined_hpx_t = torch.cat([var_hpx_t, forcing_hpx_t], dim=1)

    ua_faces = _to_hpx_faces(combined_hpx_t)
    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=_state.var_nlev)
    snapshot = trainer.prepare_snapshot(ua_faces, unix_seconds)

    horizon = _sample_horizon()

    if _state.pending_example is None:
        # First effective timestep: store source snapshot and wait for target.
        _state.pending_example = _enqueue_snapshot(snapshot, current_step, horizon=horizon)
    else:
        # Due step: train on (source, target) pair, then re-enqueue current as source.
        result = trainer.train_step(_state.pending_example.source_snapshot, snapshot)
        comin.print_info(
            f"[rank={rank}] step={current_step} loss={result['loss']:.6f} "
        )
        _state.pending_example = _enqueue_snapshot(snapshot, current_step, horizon=horizon)
