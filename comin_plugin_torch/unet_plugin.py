import comin
import datetime
import os
import socket
import sys
from dataclasses import dataclass
from typing import List, Optional

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
    Reduction,
)

try:
    _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may be undefined when COMIN loads the plugin via exec().
    _PLUGIN_DIR = os.environ.get("MESSE_PLUGIN_DIR", os.getcwd())
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from unet_online import UNetSnapshot, OnlineUNetTrainer

DOMAIN_ID = int(os.environ.get("MESSE_DOMAIN_ID", "1"))
ICON_VARIABLE_NAME = os.environ.get("MESSE_ICON_VAR", "u")
HPX_LEVEL = int(os.environ.get("MESSE_HPX_LEVEL", "7"))
MAX_FORECAST_HORIZON = int(os.environ.get("MESSE_FORECAST_HORIZON", "90"))
EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "unet_online.pt")
SAVE_INTERVAL_SECONDS: int = (
    86400  # P1D — save the model the same frequency as icon output
)

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
# Non-GPU ranks never enter NCCL or the online UNet trainer.
# ----------------------------------------------------------------------------
_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# A single digit means exactly one GPU was assigned to this rank.
_has_physical_gpu = _cuda_vis.isdigit()

# Cap GPU ranks to one per HEALPix face (12 base faces).
# MPI_Scan gives the inclusive prefix sum so we can determine each GPU rank's
# 0-based index among all GPU ranks without a second comm.Split.
# Only the first 12 GPU ranks join DDP so each owns exactly one complete face
# (nside² cells). Extra GPU ranks beyond 12 fall through as non-GPU ranks.
_gpu_prefix_sum = comm.scan(1 if _has_physical_gpu else 0, op=MPI.SUM)
_raw_gpu_rank = (_gpu_prefix_sum - 1) if _has_physical_gpu else -1
has_gpu = _has_physical_gpu and _raw_gpu_rank < 12

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

# Only computing ranks (one per HEALPix face) join the healpix_target component.
# Non-computing ranks do not join this component, so YAC will not wait for them
# to call put/get on HEALPix fields.  All ranks remain in source_comp (ICON grid)
# and must still participate in the ICON-side YAC exchanges.
nside = 2**HPX_LEVEL
total_hp_cells = 12 * 4**HPX_LEVEL
if has_gpu:
    target_comp = yac.predef_comp("healpix_target")
    cells_per_rank = total_hp_cells // num_calculate_processes
    start_idx = compute_rank * cells_per_rank
    # Last GPU rank takes the remainder
    end_idx = (
        (compute_rank + 1) * cells_per_rank
        if compute_rank != num_calculate_processes - 1
        else total_hp_cells
    )
    local_hp_indices = np.arange(start_idx, end_idx)
else:
    target_comp = None
    start_idx = 0
    end_idx = 0


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


if has_gpu:
    hp_grid, hp_points = _make_healpix_grid(
        "hp_level6_grid", nside, cell_idx=local_hp_indices
    )
else:
    hp_points = None


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


def _insert_icon_cells(pred_cells: np.ndarray, buffer) -> None:
    """Scatter (ncells, nlev) float64 array into COMIN (nproma, nlev, nblk) buffer in-place.

    Inverse of _extract_icon_cells: cell c maps to buf[c % nproma, :, c // nproma].
    """
    nc, nlev = pred_cells.shape
    buf = xp.asarray(buffer)
    while buf.ndim > 3 and buf.shape[-1] == 1:
        buf = buf[..., 0]
    nproma_val = buf.shape[0]
    c = np.arange(nc)
    buf[c % nproma_val, :, c // nproma_val] = xp.asarray(pred_cells)


def _to_hpx_faces(owned_vals: torch.Tensor) -> torch.Tensor:
    """Reshape (n_owned_pixels, nlev) to (faces_per_rank, nlev, nside, nside)."""
    nside = 2**HPX_LEVEL
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)
    return (
        owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()
    )


def _from_hpx_faces(pred_faces: torch.Tensor) -> torch.Tensor:
    """Reshape (faces_per_rank, nlev, nside, nside) back to (n_owned_pixels, nlev).

    Inverse of _to_hpx_faces.
    """
    return pred_faces.permute(0, 2, 3, 1).contiguous().reshape(-1, pred_faces.shape[1])


# ----------------------------------------------------------------------------
# Plugin state — all mutable runtime state lives here instead of as globals.
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        "current_step",
        "pending_example",
        "trainer",
        "icon_var",
        "AI_pred",
        "hp_field_src",
        "hp_field_tgt",
        "pred_hp_src",
        "pred_icon_tgt",
        "var_nlev_groups",
        "owned_face_ids",
        "step_len_seconds",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.pending_example: Optional["ForecastExample"] = None
        self.trainer: Optional[OnlineUNetTrainer] = None
        self.icon_var = None  # COMIN variable handle for ua, set in sec_ctor
        self.AI_pred = None  # COMIN variable handle for predicted ua, set in sec_ctor
        self.hp_field_src: Optional[Field] = None  # YAC field for ua on ICON grid
        self.hp_field_tgt: Optional[Field] = None  # YAC field for ua on HEALPix grid
        self.pred_hp_src: Optional[Field] = (
            None  # YAC field for prediction on HEALPix (reverse coupling source)
        )
        self.pred_icon_tgt: Optional[Field] = (
            None  # YAC field for prediction on ICON (reverse coupling target)
        )
        self.var_nlev_groups: Optional[List[int]] = (
            None  # [ua_nlev] = [90], set in setup_coupling
        )
        self.owned_face_ids: Optional[torch.Tensor] = (
            None  # HEALPix face indices owned by this rank
        )
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
    source_snapshot: UNetSnapshot
    horizon: int
    due_step: int
    mean: torch.Tensor  # Per-level mean for denormalization
    std: torch.Tensor  # Per-level std for denormalization


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


def _get_trainer(nlev: int) -> OnlineUNetTrainer:
    if _state.trainer is not None:
        if _state.trainer.nlev != nlev:
            raise RuntimeError(
                f"ICON level count changed: {_state.trainer.nlev} → {nlev}"
            )
        return _state.trainer

    _trainer_kwargs = dict(
        nlev=nlev,
        lr=float(os.environ.get("MESSE_UNET_LR", "2e-4")),
        model_channels=int(os.environ.get("MESSE_UNET_MODEL_CH", "64")),
        grad_clip=1.0,
        use_ddp=dist.is_initialized(),
        device=torch.device("cuda", 0),
        log_fn=comin.print_info,
        rank=rank,
    )

    if os.path.exists(CHECKPOINT_PATH):
        # Checkpoint found: restore model and optimizer state from previous run.
        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location=torch.device("cuda", 0),
            weights_only=False,
        )
        _state.trainer = OnlineUNetTrainer(**_trainer_kwargs)
        _state.trainer.model.load_state_dict(ckpt["model_state_dict"])
        _state.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        saved_step = ckpt.get("step", "unknown")
        _state.current_step = int(saved_step) if isinstance(saved_step, int) else 0
        comin.print_info(
            f"[rank={rank}] Restored UNet from checkpoint {CHECKPOINT_PATH} "
            f"(step={saved_step}), resuming from step={_state.current_step}"
        )
    else:
        # No checkpoint: fresh initialization.
        _state.trainer = OnlineUNetTrainer(**_trainer_kwargs)
        comin.print_info(f"[rank={rank}] UNet trainer initialized fresh: nlev={nlev}")

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
    snapshot: UNetSnapshot,
    current_step: int,
    horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> ForecastExample:
    due_step = current_step + horizon
    comin.print_info(
        f"[rank={rank}] enqueued snapshot at step={current_step}, due={due_step}"
    )
    return ForecastExample(
        source_snapshot=snapshot,
        horizon=horizon,
        due_step=due_step,
        mean=mean,
        std=std,
    )


# ----------------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------------


def _save_checkpoint(trainer: OnlineUNetTrainer, step: int) -> None:
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


def _rollback_checkpoint(trainer: OnlineUNetTrainer) -> int:
    """Reload model + optimizer from the last saved checkpoint on all GPU ranks.

    Returns the step stored in the checkpoint, or 0 if no checkpoint exists.
    All compute ranks reload the same file so weights and optimizer moments stay
    consistent after a NaN-triggered recovery.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        comin.print_info(
            f"[rank={rank}] No checkpoint to roll back to — weights remain corrupted"
        )
        return 0
    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location=torch.device("cuda", 0),
        weights_only=False,
    )
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = int(ckpt.get("step", 0))
    comin.print_info(
        f"[rank={rank}] Rolled back to checkpoint at step={step} from {CHECKPOINT_PATH}"
    )
    return step


# ----------------------------------------------------------------------------
# COMIN callbacks
# ----------------------------------------------------------------------------

# primary constructor to save prediction

var_descriptor = ("var_predict", DOMAIN_ID)
comin.var_request_add(var_descriptor, lmodexclusive=True)
comin.metadata_set(var_descriptor, zaxis_id=comin.COMIN_ZAXIS_3D)


# secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    # the variable to read
    _state.icon_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (ICON_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )

    # the prediction to save
    _state.AI_pred = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        ("var_predict", DOMAIN_ID),
        comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
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

    # Store group depths: group 0 = ua (3D)
    _state.var_nlev_groups = [var_nlev]

    comin.print_info(f"[rank={rank}] timestep length: {step_len} seconds")
    comin.print_info(
        f"[rank={rank}] ua nlev={var_nlev}, groups={_state.var_nlev_groups}"
    )

    # ICON source field: all ranks define this (they all own a slice of ICON cells).
    _state.hp_field_src = Field.create(
        "var_remap", source_comp, icon_cell_centers, var_nlev, step_len, TimeUnit.SECOND
    )
    # HEALPix target field: only compute ranks join healpix_target component.
    if has_gpu:
        _state.hp_field_tgt = Field.create(
            "var_target", target_comp, hp_points, var_nlev, step_len, TimeUnit.SECOND
        )

    interp = InterpolationStack()
    interp.add_nnn(NNNReductionType.AVG, n=1)

    # def_couple is called by all ranks; YAC resolves coupling per component membership.
    yac.def_couple(
        "icon_r2b4_source",
        "icon_grid",
        "var_remap",
        "healpix_target",
        "hp_level6_grid",
        "var_target",
        step_len,
        TimeUnit.SECOND,
        Reduction.TIME_NONE,
        interp,
    )
    # Reverse coupling: HEALPix prediction -> ICON grid (mirrors forward ua coupling)
    if has_gpu:
        _state.pred_hp_src = Field.create(
            "pred_hp_src", target_comp, hp_points, var_nlev, step_len, TimeUnit.SECOND
        )
    # All ICON ranks receive the prediction to scatter back to their local ICON cells.
    _state.pred_icon_tgt = Field.create(
        "pred_icon_tgt",
        source_comp,
        icon_cell_centers,
        var_nlev,
        step_len,
        TimeUnit.SECOND,
    )
    interp_rev = InterpolationStack()
    interp_rev.add_nnn(NNNReductionType.AVG, n=1)
    yac.def_couple(
        "healpix_target",
        "hp_level6_grid",
        "pred_hp_src",
        "icon_r2b4_source",
        "icon_grid",
        "pred_icon_tgt",
        step_len,
        TimeUnit.SECOND,
        Reduction.TIME_NONE,
        interp_rev,
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online UNet training callback called by ICON at each time step."""

    # ------------------------------------------------------------------
    # STEP 1 — All ICON ranks push their local cell data to the forward
    # YAC coupling.  Every rank owns a slice of the ICON grid, so all
    # ranks must contribute; otherwise the interpolation to HEALPix is
    # incomplete and YAC stalls.
    # ------------------------------------------------------------------
    icon_var_cells = _extract_icon_cells(_state.icon_var)  # (ncells, nlev_ua)
    icon_var_cells_np = (
        icon_var_cells.get() if hasattr(icon_var_cells, "get") else icon_var_cells
    )
    _state.hp_field_src.put(icon_var_cells_np)

    if not has_gpu:
        # Non-compute ranks: collect the HEALPix→ICON prediction written by
        # the GPU ranks and scatter it to their local ICON cells, then return.
        pred_icon_np, _ = _state.pred_icon_tgt.get()
        _insert_icon_cells(pred_icon_np.T.astype(np.float64), _state.AI_pred)
        return

    # ------------------------------------------------------------------
    # GPU / compute ranks from here.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # STEP 2 — GPU ranks receive the HEALPix-interpolated field from YAC.
    # This is called at every step (no early-return during waiting) so the
    # reverse coupling is never left in a stalled state.
    # ------------------------------------------------------------------
    var_hpx, info = _state.hp_field_tgt.get()

    # Convert numpy/cupy arrays from YAC to float32 GPU tensors of shape (n_pixels, nlev).
    # YAC returns (nlev, ncells) for multi-level fields and (ncells,) for single-level.
    var_hpx_t = torch.as_tensor(xp.asarray(var_hpx), device="cuda").float()
    if var_hpx_t.ndim == 1:
        var_hpx_t = var_hpx_t.unsqueeze(-1)
    elif var_hpx_t.ndim == 2:
        var_hpx_t = var_hpx_t.T  # (nlev, ncells) -> (ncells, nlev)

    ua_faces = _to_hpx_faces(var_hpx_t)  # (faces, nlev_ua, nside, nside)

    mean = ua_faces.mean(dim=(0, 2, 3), keepdim=True)
    std = ua_faces.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    ua_faces_norm = (ua_faces - mean) / std

    if not torch.isfinite(ua_faces_norm).all():
        comin.print_info(
            f"[rank={rank}] step={current_step} WARNING: NaN/Inf in ua_faces_norm "
            f"(ua_faces min={ua_faces.min().item():.3f} max={ua_faces.max().item():.3f} "
            f"std_min={std.min().item():.3e})"
        )

    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=_state.var_nlev_groups[0])
    snapshot = trainer.prepare_snapshot(ua_faces_norm, unix_seconds)

    # horizon = _sample_horizon()
    horizon = 10  # for debugging, use a fixed horizon of 10 steps

    # ------------------------------------------------------------------
    # STEP 3 — Training logic: only train when the due step is reached.
    # ------------------------------------------------------------------
    if _state.pending_example is None:
        # First effective timestep: store source snapshot and wait for target.
        _state.pending_example = _enqueue_snapshot(
            snapshot, current_step, horizon=horizon, mean=mean, std=std
        )
    elif current_step >= _state.pending_example.due_step:
        # Due step: train on (source, target) pair, then re-enqueue current as source.
        comin.print_info(
            f"[rank={rank}] step={current_step} ua_faces: "
            f"min={ua_faces.min().item():.3f} max={ua_faces.max().item():.3f} "
            f"mean={ua_faces.mean().item():.3f} std={ua_faces.std().item():.3f}"
        )
        result = trainer.train_step(_state.pending_example.source_snapshot, snapshot)
        comin.print_info(
            f"[rank={rank}] step={current_step} loss={result['loss']:.6f} "
            f"grad_norm={result.get('grad_norm', 0.0):.4f} "
            f"skipped={result.get('skipped', False)}"
        )
        if result.get("needs_rollback"):
            comin.print_info(
                f"[rank={rank}] step={current_step} NaN detected — rolling back to last checkpoint"
            )
            _rollback_checkpoint(_state.trainer)
            _state.pending_example = None  # discard potentially corrupted snapshot
        else:
            _state.pending_example = _enqueue_snapshot(
                snapshot, current_step, horizon=horizon, mean=mean, std=std
            )
    else:
        # Waiting for the due step: log only; inference still runs below.
        comin.print_info(
            f"[rank={rank}] step={current_step} waiting "
            f"horizon={_state.pending_example.horizon} "
            f"(due={_state.pending_example.due_step})"
        )

    # ------------------------------------------------------------------
    # STEP 4 — Run inference at every step and push prediction to ICON.
    # Inference during waiting steps keeps the reverse YAC coupling alive
    # across all time steps so non-GPU ranks never stall.
    # ------------------------------------------------------------------
    pred_faces = trainer.predict(snapshot)  # (F, nlev, nside, nside) GPU float32
    # Denormalize prediction: original = normalized * std + mean
    pred_faces_denorm = pred_faces * std + mean
    pred_flat_np = (
        _from_hpx_faces(pred_faces_denorm).cpu().numpy()
    )  # (n_local_hp_cells, nlev) float32
    _state.pred_hp_src.put(pred_flat_np)  # HEALPix source → YAC
    pred_icon_np, _ = _state.pred_icon_tgt.get()  # (nlev, ncells) on ICON grid
    _insert_icon_cells(pred_icon_np.T.astype(np.float64), _state.AI_pred)


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """Cleanly tear down PyTorch distributed before MPI_Finalize.

    Without this, the NCCL process group holds onto MPI communicators and
    causes UCX 'unexpected tag-receive descriptor' warnings that prevent
    ICON from writing finish.status, breaking the restart chain.
    """
    if has_gpu and dist.is_initialized():
        dist.destroy_process_group()
        comin.print_info(f"[rank={rank}] PyTorch distributed destroyed")
