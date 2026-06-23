import comin
import datetime
import os
import socket
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

from unet_online import UNetSnapshot, OnlineUNetTrainer
from utils import (
    setup_mpi_dist,
    setup_icon_grid,
    setup_hpx_grid,
    parse_icon_datetime,
    to_hpx_faces,
    from_hpx_faces,
    save_checkpoint,
    rollback_checkpoint,
    sample_horizon,
    extract_icon_cells,
    insert_icon_cells,
    enqueue_snapshot,
    ForecastExample,
)

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


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DOMAIN_ID = int(os.environ.get("MESSE_DOMAIN_ID", "1"))
ICON_VARIABLE_NAME = os.environ.get("MESSE_ICON_VAR", "u_10m")
HPX_LEVEL = int(os.environ.get("MESSE_HPX_LEVEL", "6"))
MAX_FORECAST_HORIZON = int(os.environ.get("MESSE_FORECAST_HORIZON", "4"))  # steps
EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "unet_online.pt")
SNAPSHOTS_DIR = os.path.join(EXPERIMENTS_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
DRY_RUN_TIME_SECONDS: int = 2592000  # 1 month
SAVE_INTERVAL_SECONDS: int = 86400  # 1 day


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
# MPI and PyTorch distributed setup
# ----------------------------------------------------------------------------

# compute ranks on GPU-bearing MPI ranks, None on IO ranks without GPU
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
size = comm.Get_size()
rank = comm.Get_rank()

# GPU ranks
compute_comm, compute_rank, compute_size, has_gpu = setup_mpi_dist(
    comm
)  # all None for io ranks (without GPU)
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)


# ----------------------------------------------------------------------------
# YAC setup (HEALPix interpolation)
# ----------------------------------------------------------------------------
domain = comin.descrdata_get_domain(DOMAIN_ID)
assert glob.yac_instance_id != -1, "The host-model is not configured with yac"

yac = YAC.from_id(glob.yac_instance_id)
source_comp, icon_grid, icon_cell_centers = setup_icon_grid(yac, glob, domain, has_gpu)
target_comp, hp_grid, hp_points = setup_hpx_grid(
    yac, HPX_LEVEL, compute_rank, compute_size, has_gpu
)


# ----------------------------------------------------------------------------
# Plugin state
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        # info
        "current_step",
        "nlev",
        "step_len_seconds",
        "horizon_length_steps",
        # data
        "icon_var",
        "AI_var",
        # YAC ICON-> HEALPix
        "hp_field_src",
        "hp_field_tgt",
        # YAC HEALPix -> ICON (reverse coupling for prediction)
        "pred_hp_src",
        "pred_icon_tgt",
        # dry run for normalization
        "dryrun_time",
        "dryrun_done",
        "accum_count",
        "hp_accum_sum",
        "hp_accum_sumsq",
        "hp_accum_mean",
        "hp_accum_std",
        # training
        "pending_example",
        "trainer",
    )

    def __init__(self) -> None:
        # info
        self.current_step: int = 0
        self.nlev: Optional[int] = None  # Number of vertical levels
        self.step_len_seconds: Optional[int] = None  # ICON timestep length in seconds
        self.horizon_length_steps: Optional[int] = (
            None  # Forecast horizon length in steps
        )

        # data
        self.icon_var = None  # COMIN variable handle for ua, set in sec_ctor
        self.AI_var = None  # COMIN variable handle for predicted ua, set in sec_ctor

        # YAC
        self.hp_field_src: Optional[Field] = None  # YAC field for ua on ICON grid
        self.hp_field_tgt: Optional[Field] = None  # YAC field for ua on HEALPix grid
        self.pred_hp_src: Optional[Field] = (
            None  # YAC field for prediction on HEALPix (reverse coupling source)
        )
        self.pred_icon_tgt: Optional[Field] = (
            None  # YAC field for prediction on ICON (reverse coupling target)
        )

        # normalization dry run
        self.dryrun_time: Optional[int] = None  # Time for dry run in seconds
        self.dryrun_done: bool = False  # Whether the dry run is complete
        self.accum_count: int = 0
        self.hp_accum_sum: Optional[torch.Tensor] = None
        self.hp_accum_sumsq: Optional[torch.Tensor] = None
        self.hp_accum_mean: Optional[torch.Tensor] = (
            None  # Accumulated mean for HEALPix normalization during dry run
        )
        self.hp_accum_std: Optional[torch.Tensor] = (
            None  # Accumulated std for HEALPix normalization during dry run
        )

        # training
        self.pending_example: Optional["ForecastExample"] = None
        self.trainer: Optional[OnlineUNetTrainer] = None


_state = _State()


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


# Time helper
def _icon_time_unix_seconds() -> float:
    return float(parse_icon_datetime(comin.current_get_datetime()).timestamp())




# Trainer helpers
def _get_trainer(nlev: int) -> OnlineUNetTrainer:
    if _state.trainer is not None:
        return _state.trainer

    # args
    _trainer_kwargs = dict(
        nlev=nlev,
        lr=float(os.environ.get("MESSE_UNET_LR", "2e-4")),
        model_channels=int(os.environ.get("MESSE_UNET_MODEL_CH", "64")),
        grad_clip=1.0,
        use_ddp=dist.is_initialized(),
        device=torch.device("cuda", 0),
        # for info
        log_fn=comin.print_info,
        rank=rank,
    )

    # checkpoint restore
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


# ----------------------------------------------------------------------------
# COMIN callbacks
# ----------------------------------------------------------------------------

# primary constructor to save prediction
var_descriptor = ("var_predict", DOMAIN_ID)
comin.var_request_add(var_descriptor, lmodexclusive=True)
comin.metadata_set(var_descriptor, zaxis_id=comin.COMIN_ZAXIS_2D)


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
    _state.AI_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        ("var_predict", DOMAIN_ID),
        comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    )


@comin.register_callback(comin.EP_ATM_YAC_DEFCOMP_AFTER)
def setup_coupling():
    # Non-GPU (IO) ranks are not part of the healpix_target component and
    # have no HEALPix grid objects — skip all YAC field/coupling definitions.
    if not has_gpu:
        return

    # integration rate
    step_len = str(int(comin.descrdata_get_timesteplength(1)))
    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    # COMIN uses
    #   (nproma, nlev, nblk) for 3-D fields
    #   (nproma, nblk) for surface fields.

    _icon_arr = xp.asarray(_state.icon_var)
    while _icon_arr.ndim > 3 and _icon_arr.shape[-1] == 1:
        _icon_arr = _icon_arr[..., 0]
    var_nlev = int(_icon_arr.shape[1]) if _icon_arr.ndim >= 3 else 1
    _state.nlev = var_nlev  # Store nlev in state for trainer initialization

    comin.print_info(f"[rank={rank}] timestep length: {step_len} seconds")

    # from icon grid to healpix grid
    _state.hp_field_src = Field.create(
        "var_remap", source_comp, icon_cell_centers, var_nlev, step_len, TimeUnit.SECOND
    )
    _state.hp_field_tgt = Field.create(
        "var_target", target_comp, hp_points, var_nlev, step_len, TimeUnit.SECOND
    )

    interp = InterpolationStack()
    interp.add_nnn(NNNReductionType.AVG, n=1)

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
    _state.pred_hp_src = Field.create(
        "pred_hp_src", target_comp, hp_points, var_nlev, step_len, TimeUnit.SECOND
    )
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
def dry_run():
    """Dry run to accumulate mean and std over the first `DRY_RUN_TIME_SECONDS` seconds."""

    # integration rate
    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    if not has_gpu:
        return

    if _state.dryrun_done:
        return

    dry_run_steps = DRY_RUN_TIME_SECONDS // _state.step_len_seconds
    current_step = _state.current_step

    if current_step < dry_run_steps:
        icon_var_cells = extract_icon_cells(
            _state.icon_var, domain.cells.ncells
        )  # (ncells, nlev), xp array

        # to healpix
        icon_var_cells_np = (
            icon_var_cells.get() if hasattr(icon_var_cells, "get") else icon_var_cells
        )

        _state.hp_field_src.put(icon_var_cells_np)
        var_hpx, info = _state.hp_field_tgt.get()

        # Apply the same shape transformation as in training so that mean/std
        # have shape (faces, nlev, nside, nside), matching ua_faces exactly.
        var_hpx_t = torch.as_tensor(xp.asarray(var_hpx), device="cuda").float()
        if var_hpx_t.ndim == 1:
            var_hpx_t = var_hpx_t.unsqueeze(-1)
        elif var_hpx_t.ndim == 2:
            var_hpx_t = var_hpx_t.T  # (nlev, ncells) -> (ncells, nlev)
        ua_faces = to_hpx_faces(var_hpx_t, HPX_LEVEL)  # (faces, nlev, nside, nside)
        ua_faces_d = ua_faces.double()  # float64 for stable accumulation

        if _state.hp_accum_sum is None:
            _state.hp_accum_sum = torch.zeros_like(ua_faces_d)
            _state.hp_accum_sumsq = torch.zeros_like(ua_faces_d)

        _state.hp_accum_sum += ua_faces_d
        _state.hp_accum_sumsq += ua_faces_d**2

        _state.accum_count += 1
        _state.current_step += 1

        comin.print_info(
            f"[rank={rank}] Dry run step {current_step + 1}/{dry_run_steps}"
        )
        return  # don't fall through to finalization on the same call

    # current_step >= dry_run_steps: finalize
    mean = (_state.hp_accum_sum / _state.accum_count).float()
    e_x2 = (_state.hp_accum_sumsq / _state.accum_count).float()
    var = torch.clamp(e_x2 - mean**2, min=1e-6)
    std = torch.sqrt(var)

    _state.hp_accum_mean = mean  # (faces, nlev, nside, nside) float32 CUDA tensor
    _state.hp_accum_std = std
    _state.dryrun_done = True

    comin.print_info(
        f"[rank={rank}] Dry run complete after {_state.accum_count} steps: "
        f"mean range=[{float(mean.min())}, {float(mean.max())}], "
        f"std range=[{float(std.min())}, {float(std.max())}]"
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online UNet training callback called by ICON at each time step."""

    if not has_gpu:
        return

    if not _state.dryrun_done:
        comin.print_info(f"[rank={rank}] Dry run not complete, skipping training")
        return

    current_step = _state.current_step
    _state.current_step += 1

    # Periodic checkpoint save
    if (
        _state.trainer is not None
        and _state.step_len_seconds is not None
        and current_step > 0
    ):
        steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
        if current_step % steps_per_save == 0:
            save_checkpoint(_state.trainer, CHECKPOINT_PATH, compute_rank, current_step)
            comin.print_info(f"[rank={rank}] Checkpoint saved at step={current_step}")

    # Mode: Waiting; between t and t+horizon
    if (
        _state.pending_example is not None
        and current_step < _state.pending_example.due_step
    ):
        comin.print_info(
            f"[rank={rank}] step={current_step} waiting "
            f"horizon={_state.pending_example.horizon} "
            f"(due={_state.pending_example.due_step})"
        )
        return

    icon_var_cells = extract_icon_cells(_state.icon_var, domain.cells.ncells)  # (shape: ncells, nlev_ua; nc)

    # Interpolation from native ICON grid to HEALPix using YAC
    # currently YAC only supports numpy arrays, wait for update to support cupy arrays
    icon_var_cells_np = (
        icon_var_cells.get() if hasattr(icon_var_cells, "get") else icon_var_cells
    )

    _state.hp_field_src.put(icon_var_cells_np)
    var_hpx, info = _state.hp_field_tgt.get()

    # Convert numpy/cupy arrays from YAC to float32 GPU tensors of shape (n_pixels, nlev).
    # YAC returns (nlev, ncells) for multi-level fields and (ncells,) for single-level.
    var_hpx_t = torch.as_tensor(xp.asarray(var_hpx), device="cuda").float()
    if var_hpx_t.ndim == 1:
        var_hpx_t = var_hpx_t.unsqueeze(-1)
    elif var_hpx_t.ndim == 2:
        var_hpx_t = var_hpx_t.T  # (nlev, ncells) -> (ncells, nlev)

    ua_faces = to_hpx_faces(var_hpx_t, HPX_LEVEL)  # (faces, nlev_ua, nside, nside)

    # normalize
    mean = _state.hp_accum_mean
    std = _state.hp_accum_std
    ua_faces_norm = (ua_faces - mean) / std

    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=_state.nlev)
    snapshot = trainer.prepare_snapshot(
        ua_faces_norm, unix_seconds
    )  # the time is not used in current UNet

    # horizon = _sample_horizon()
    horizon = MAX_FORECAST_HORIZON  # for debugging, use a fixed horizon of 10 steps
    _state.horizon_length_steps = horizon

    if _state.pending_example is None:
        # First effective timestep: store source snapshot and wait for target.
        _state.pending_example = enqueue_snapshot(
            snapshot, current_step, horizon=horizon, mean=mean, std=std
        )
    else:
        # Due step: train on (source, target) pair, then re-enqueue current as source.
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
            rollback_checkpoint(_state.trainer, CHECKPOINT_PATH)
            _state.pending_example = None  # discard potentially corrupted snapshot
        else:
            _state.pending_example = enqueue_snapshot(
                snapshot, current_step, horizon=horizon, mean=mean, std=std
            )

    # Run inference and scatter prediction to ICON grid via reverse YAC coupling
    pred_faces = trainer.predict(snapshot)  # (F, nlev, nside, nside) GPU float32
    # Denormalize prediction: original = normalized * std + mean
    pred_faces_denorm = pred_faces * std + mean

    # # Save (input, prediction) snapshot pair at for debugging
    # if _state.step_len_seconds is not None and current_step > 0:
    #     _steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
    #     if current_step % _steps_per_save == 0:
    #         _save_snapshot_pair(ua_faces, pred_faces_denorm, current_step)

    pred_flat_np = (
        from_hpx_faces(pred_faces_denorm).cpu().numpy()
    )  # (n_local_hp_cells, nlev) float32
    _state.pred_hp_src.put(pred_flat_np)  # HEALPix source → YAC
    pred_icon_np, _ = _state.pred_icon_tgt.get()  # (nlev, ncells) on ICON grid
    insert_icon_cells(pred_icon_np.T.astype(np.float64), _state.AI_var)


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
