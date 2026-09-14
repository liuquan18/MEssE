import comin
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from mpi4py import MPI

try:
    _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may be undefined when COMIN loads the plugin via exec().
    _PLUGIN_DIR = os.environ.get("MESSE_PLUGIN_DIR", os.getcwd())
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from icon_mgrid_utils import LocalMGrid, build_local_mgrid, load_coarse_grid, load_parent_index
from fieldspace_online import FieldSpaceSnapshot, OnlineFieldSpaceTrainer
from MEssE.utils.icon_online_helper import (
    RunningMeanStd,
    RolloutBuffer,
    setup_mpi_dist,
    parse_icon_datetime,
    save_checkpoint,
    rollback_checkpoint,
    extract_icon_cells,
    insert_icon_cells,
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DOMAIN_ID = int(os.environ.get("MESSE_DOMAIN_ID", "1"))
ICON_VARIABLE_NAME = os.environ.get("MESSE_ICON_VAR", "u_10m")
ROLLOUT_STEPS = int(os.environ.get("MESSE_FS_ROLLOUT_STEPS", "4"))  # forecast horizon, in steps
ATT_DIM = int(os.environ.get("MESSE_FS_ATT_DIM", "32"))
N_HEAD_CHANNELS = int(os.environ.get("MESSE_FS_N_HEAD_CHANNELS", "8"))

EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "fieldspace_online.pt")
DRY_RUN_TIME_SECONDS: int = 86400  # 1 day
SAVE_INTERVAL_SECONDS: int = 86400  # 1 day

# Paths to the coarse and fine grids
COARSE_GRID_PATH = os.environ.get(
    "MESSE_FS_COARSE_GRID_PATH",
    "/pool/data/ICON/grids/public/edzw/icon_grid_0011_R02B03_R.nc",
)
FINE_GRID_PATH = os.environ.get(
    "MESSE_FS_FINE_GRID_PATH",
    "/pool/data/ICON/grids/public/edzw/icon_grid_0012_R02B04_G.nc",
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
# MPI and PyTorch distributed setup (DDP)
# ----------------------------------------------------------------------------

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
size = comm.Get_size()
rank = comm.Get_rank()

# GPU ranks and DDP world
compute_comm, compute_rank, compute_size, has_gpu = setup_mpi_dist(comm)
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)


# ----------------------------------------------------------------------------
# Native ICON grid + local two-zoom multi-grid (no YAC, no HEALPix)
# ----------------------------------------------------------------------------
domain = comin.descrdata_get_domain(DOMAIN_ID)
comin.print_info(
    f"[rank={rank}] domain.grid_filename={getattr(domain, 'grid_filename', '<unavailable>')} "
    f"(expected to match FINE_GRID_PATH={FINE_GRID_PATH})"
)

_coarse_grid = None
_parent_index_global: Optional[np.ndarray] = None
_mgrid: Optional[LocalMGrid] = None
if has_gpu:
    _coarse_grid = load_coarse_grid(COARSE_GRID_PATH)
    _parent_index_global = load_parent_index(FINE_GRID_PATH)
    _mgrid = build_local_mgrid(
        domain,
        nproma=glob.nproma,
        coarse_grid=_coarse_grid,
        parent_index_global=_parent_index_global,
        device=torch.device("cuda", 0),
    )
    comin.print_info(
        f"[rank={rank}] local mgrid: n_valid_fine={_mgrid.n_valid_fine} "
        f"n_fine_kept={_mgrid.n_fine_kept} n_coarse={_mgrid.n_coarse} "
        f"kept_fraction={_mgrid.n_fine_kept / max(_mgrid.n_valid_fine, 1):.3f} "
        f"group_sizes={_mgrid.parent_group_size_histogram}"
    )


# ----------------------------------------------------------------------------
# Plugin state
# ----------------------------------------------------------------------------
class _State:
    __slots__ = (
        "current_step",
        "nlev",
        "step_len_seconds",
        # data
        "icon_var",
        "AI_var",
        # dry run for normalization (see icon_online_helper.RunningMeanStd)
        "normalizer",
        # source snapshot + fixed-horizon scheduling (ROLLOUT_STEPS steps
        # ahead); see icon_online_helper.RolloutBuffer
        "rollout_pair",
        "trainer",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.nlev: Optional[int] = None
        self.step_len_seconds: Optional[int] = None

        self.icon_var = None
        self.AI_var = None

        self.normalizer: Optional[RunningMeanStd] = None
        self.rollout_pair: RolloutBuffer = RolloutBuffer(ROLLOUT_STEPS)
        self.trainer: Optional[OnlineFieldSpaceTrainer] = None


_state = _State()


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _icon_time_unix_seconds() -> float:
    return float(parse_icon_datetime(comin.current_get_datetime()).timestamp())


def _get_trainer(nlev: int) -> OnlineFieldSpaceTrainer:
    if _state.trainer is not None:
        return _state.trainer

    _trainer_kwargs = dict(
        nlev=nlev,
        mgrid=_mgrid,
        lr=float(os.environ.get("MESSE_FS_LR", "2e-4")),
        att_dim=ATT_DIM,
        n_head_channels=N_HEAD_CHANNELS,
        grad_clip=1.0,
        use_ddp=dist.is_initialized(),
        device=torch.device("cuda", 0),
        log_fn=comin.print_info,
        rank=rank,
    )

    if os.path.exists(CHECKPOINT_PATH):
        _state.trainer = OnlineFieldSpaceTrainer(**_trainer_kwargs)
        if _state.trainer.model is not None:
            ckpt = torch.load(
                CHECKPOINT_PATH,
                map_location=torch.device("cuda", 0),
                weights_only=False,
            )
            _state.trainer.model.load_state_dict(ckpt["model_state_dict"])
            _state.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            saved_step = ckpt.get("step", "unknown")
            _state.current_step = int(saved_step) if isinstance(saved_step, int) else 0
            comin.print_info(
                f"[rank={rank}] Restored FieldSpace model from checkpoint {CHECKPOINT_PATH} "
                f"(step={saved_step}), resuming from step={_state.current_step}"
            )
    else:
        _state.trainer = OnlineFieldSpaceTrainer(**_trainer_kwargs)
        comin.print_info(f"[rank={rank}] FieldSpace trainer initialized fresh: nlev={nlev}")

    return _state.trainer


# ----------------------------------------------------------------------------
# COMIN callbacks
# ----------------------------------------------------------------------------

var_descriptor = ("var_predict", DOMAIN_ID)
comin.var_request_add(var_descriptor, lmodexclusive=True)
comin.metadata_set(var_descriptor, zaxis_id=comin.COMIN_ZAXIS_2D)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    _state.icon_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        (ICON_VARIABLE_NAME, DOMAIN_ID),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )
    _state.AI_var = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        ("var_predict", DOMAIN_ID),
        comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    )


def _extract_compact(comin_var) -> torch.Tensor:
    """Extract this rank's ICON cells for `comin_var`, zero-pad to the full
    local fine-node count, then gather down to the compact
    (4*n_coarse, nlev) space this plugin actually trains on -- only cells
    that are part of a complete coarse quad-group (see icon_mgrid_utils.py).
    """
    cells = extract_icon_cells(comin_var, domain.cells.ncells)  # (ncells, nlev) or (ncells,)
    t = torch.as_tensor(xp.asarray(cells), device="cuda").float()
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    full = torch.zeros(_mgrid.n_nodes_fine, t.shape[1], device="cuda")
    full[: t.shape[0]] = t
    return full[_mgrid.fine_local_ids_kept]  # (4*n_coarse, nlev)


def _maybe_save_checkpoint(current_step: int) -> None:
    if _state.trainer is None or _state.trainer.model is None:
        return
    if _state.step_len_seconds is None or current_step == 0:
        return
    steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
    if current_step % steps_per_save == 0:
        save_checkpoint(_state.trainer, CHECKPOINT_PATH, compute_rank, current_step)
        comin.print_info(f"[rank={rank}] Checkpoint saved at step={current_step}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def dry_run():
    """Dry run to estimate per-cell mean/std over the first
    `DRY_RUN_TIME_SECONDS` seconds, over the compact (4*n_coarse, nlev)
    space this plugin trains on. See icon_online_helper.RunningMeanStd."""

    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    if not has_gpu:
        return
    if _mgrid.n_coarse == 0:
        return
    if _state.normalizer is not None and _state.normalizer.done:
        return

    dry_run_steps = DRY_RUN_TIME_SECONDS // _state.step_len_seconds
    compact = _extract_compact(_state.icon_var)

    if _state.nlev is None:
        _state.nlev = int(compact.shape[1])
    if _state.normalizer is None:
        _state.normalizer = RunningMeanStd(
            shape=compact.shape, n_samples=dry_run_steps, device=torch.device("cuda", 0)
        )

    step = _state.current_step
    done = _state.normalizer.update(compact)
    _state.current_step += 1

    if done:
        stats = _state.normalizer.stats
        comin.print_info(
            f"[rank={rank}] Dry run complete after {_state.normalizer.count} steps: "
            f"mean range=[{float(stats.mean.min())}, {float(stats.mean.max())}], "
            f"std range=[{float(stats.std.min())}, {float(stats.std.max())}]"
        )
    else:
        comin.print_info(f"[rank={rank}] Dry run step {step + 1}/{dry_run_steps}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online rollout-based FieldSpaceNN training callback, called by ICON
    at each time step. Every `ROLLOUT_STEPS` steps we run one training
    update: the model rolls forward `ROLLOUT_STEPS` autoregressive steps
    from a source snapshot, and the loss is evaluated once against the
    single real snapshot observed that many steps later (see
    fieldspace_online.OnlineFieldSpaceTrainer.train_rollout_step and
    icon_online_helper.RolloutBuffer for the fixed-horizon scheduling).

    Only owned fine cells that are also part of a complete coarse
    quad-group get a prediction written back to ICON this run -- cells
    outside that compact set (this rank's `n_valid_fine - n_fine_kept`
    orphans) receive no write-back, unlike gnn_plugin.py, which writes
    back every owned cell. See the plan document's write-back note.
    """

    if not has_gpu:
        return
    if _mgrid.n_coarse == 0:
        comin.print_info(f"[rank={rank}] n_coarse == 0 on this rank, skipping training entirely")
        return
    if _state.normalizer is None or not _state.normalizer.done:
        comin.print_info(f"[rank={rank}] Dry run not complete, skipping training")
        return

    current_step = _state.current_step
    _state.current_step += 1
    _maybe_save_checkpoint(current_step)

    compact = _extract_compact(_state.icon_var)
    compact_norm = _state.normalizer.normalize(compact)

    trainer = _get_trainer(nlev=_state.nlev)
    snapshot = trainer.prepare_snapshot(compact_norm, _icon_time_unix_seconds())

    ready = _state.rollout_pair.push(snapshot)
    if ready is not None:
        source, target = ready
        result = trainer.train_rollout_step(source, target, n_steps=ROLLOUT_STEPS)
        comin.print_info(
            f"[rank={rank}] step={current_step} rollout_loss={result['loss']:.6f} "
            f"grad_norm={result.get('grad_norm', 0.0):.4f} "
            f"skipped={result.get('skipped', False)}"
        )
        if result.get("needs_rollback"):
            comin.print_info(
                f"[rank={rank}] step={current_step} NaN detected — "
                "rolling back to last checkpoint"
            )
            rollback_checkpoint(_state.trainer, CHECKPOINT_PATH)

    # Run 1-step-ahead inference and write the prediction back to ICON —
    # same grid, no reverse YAC coupling needed.
    pred = trainer.predict(snapshot, n_steps=1)  # (4*n_coarse, nlev) normalized
    pred_denorm = _state.normalizer.denormalize(pred)
    # Only write back owned cells that are also part of a complete coarse
    # quad-group: halo cells (never written back, as in gnn_plugin.py) and
    # orphan cells (excluded from the compact space entirely, see
    # icon_mgrid_utils.py) both get no write-back.
    compact_owned = _mgrid.fine_compact_owned_mask
    kept_full_ids = _mgrid.fine_local_ids_kept[compact_owned].cpu().numpy()
    pred_owned_np = pred_denorm[compact_owned].cpu().numpy()
    insert_icon_cells(pred_owned_np.astype(np.float64), _state.AI_var, indices=kept_full_ids)


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """Cleanly tear down PyTorch distributed before MPI_Finalize."""
    if has_gpu and dist.is_initialized():
        dist.destroy_process_group()
        comin.print_info(f"[rank={rank}] PyTorch distributed destroyed")
