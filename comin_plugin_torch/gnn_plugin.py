import comin
import os
import sys
from collections import deque
from typing import Deque, Optional

import numpy as np
import torch
import torch.distributed as dist

from graph_utils import build_local_graph
from gnn_online import GNNSnapshot, OnlineGNNTrainer
from utils import (
    setup_mpi_dist,
    parse_icon_datetime,
    save_checkpoint,
    rollback_checkpoint,
    extract_icon_cells,
    insert_icon_cells,
)

from mpi4py import MPI

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
#
# No YAC and no HEALPix here: the model consumes/produces the ICON native
# grid directly, so there is no regridding target resolution to configure.
# Each rank's own domain-decomposed patch (owned cells + halo ring) *is*
# the training sample/graph — see graph_utils.py for the rationale.

DOMAIN_ID = int(os.environ.get("MESSE_DOMAIN_ID", "1"))
ICON_VARIABLE_NAME = os.environ.get("MESSE_ICON_VAR", "u_10m")
ROLLOUT_STEPS = int(os.environ.get("MESSE_GNN_ROLLOUT_STEPS", "4"))  # rollout length R
HIDDEN_DIM = int(os.environ.get("MESSE_GNN_HIDDEN_DIM", "128"))
PROCESSOR_LAYERS = int(os.environ.get("MESSE_GNN_PROCESSOR_LAYERS", "6"))
EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "gnn_online.pt")
DRY_RUN_TIME_SECONDS: int = 86400  # 2592000  # 1 month
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

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
size = comm.Get_size()
rank = comm.Get_rank()

# GPU ranks. Each GPU-bearing rank owns exactly one local patch (its ICON
# domain-decomposed cells) — this *is* the DDP world: one graph sample per
# device, gradients synchronized across ranks by DDP after each rollout.
compute_comm, compute_rank, compute_size, has_gpu = setup_mpi_dist(comm)
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)


# ----------------------------------------------------------------------------
# Native ICON grid + local patch graph (no YAC, no HEALPix)
# ----------------------------------------------------------------------------
domain = comin.descrdata_get_domain(DOMAIN_ID)

_graph = None  # built lazily in sec_ctor / setup_graph, once nproma is known
if has_gpu:
    _graph = build_local_graph(
        domain, nproma=glob.nproma, device=torch.device("cuda", 0)
    )
    comin.print_info(
        f"[rank={rank}] local graph: nodes={_graph.n_nodes} "
        f"(valid={_graph.n_valid}, owned={int(_graph.owned_mask.sum())}), "
        f"edges={_graph.edge_index.shape[1]}"
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
        # dry run for normalization
        "dryrun_time",
        "dryrun_done",
        "accum_count",
        "accum_sum",
        "accum_sumsq",
        "accum_mean",
        "accum_std",
        # rollout buffer: source snapshot + up to ROLLOUT_STEPS targets
        "rollout_source",
        "rollout_targets",
        "trainer",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.nlev: Optional[int] = None
        self.step_len_seconds: Optional[int] = None

        self.icon_var = None
        self.AI_var = None

        self.dryrun_time: Optional[int] = None
        self.dryrun_done: bool = False
        self.accum_count: int = 0
        self.accum_sum: Optional[torch.Tensor] = None
        self.accum_sumsq: Optional[torch.Tensor] = None
        self.accum_mean: Optional[torch.Tensor] = None
        self.accum_std: Optional[torch.Tensor] = None

        self.rollout_source: Optional[GNNSnapshot] = None
        self.rollout_targets: Deque[GNNSnapshot] = deque(maxlen=ROLLOUT_STEPS)
        self.trainer: Optional[OnlineGNNTrainer] = None


_state = _State()


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _icon_time_unix_seconds() -> float:
    return float(parse_icon_datetime(comin.current_get_datetime()).timestamp())


def _get_trainer(nlev: int) -> OnlineGNNTrainer:
    if _state.trainer is not None:
        return _state.trainer

    _trainer_kwargs = dict(
        nlev=nlev,
        graph=_graph,
        lr=float(os.environ.get("MESSE_GNN_LR", "2e-4")),
        hidden_dim=HIDDEN_DIM,
        n_processor_layers=PROCESSOR_LAYERS,
        grad_clip=1.0,
        use_ddp=dist.is_initialized(),
        device=torch.device("cuda", 0),
        log_fn=comin.print_info,
        rank=rank,
    )

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(
            CHECKPOINT_PATH,
            map_location=torch.device("cuda", 0),
            weights_only=False,
        )
        _state.trainer = OnlineGNNTrainer(**_trainer_kwargs)
        _state.trainer.model.load_state_dict(ckpt["model_state_dict"])
        _state.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        saved_step = ckpt.get("step", "unknown")
        _state.current_step = int(saved_step) if isinstance(saved_step, int) else 0
        comin.print_info(
            f"[rank={rank}] Restored GNN from checkpoint {CHECKPOINT_PATH} "
            f"(step={saved_step}), resuming from step={_state.current_step}"
        )
    else:
        _state.trainer = OnlineGNNTrainer(**_trainer_kwargs)
        comin.print_info(f"[rank={rank}] GNN trainer initialized fresh: nlev={nlev}")

    return _state.trainer


def _reset_rollout() -> None:
    _state.rollout_source = None
    _state.rollout_targets.clear()


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


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def dry_run():
    """Dry run to accumulate per-node mean/std over the first
    `DRY_RUN_TIME_SECONDS` seconds, directly on the ICON native grid
    (no HEALPix reshaping needed)."""

    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    if not has_gpu:
        return
    if _state.dryrun_done:
        return

    dry_run_steps = DRY_RUN_TIME_SECONDS // _state.step_len_seconds
    current_step = _state.current_step

    if current_step < dry_run_steps:
        icon_var_nodes = extract_icon_cells(
            _state.icon_var, domain.cells.ncells
        )  # (ncells, nlev) or (ncells,)
        var_t = torch.as_tensor(xp.asarray(icon_var_nodes), device="cuda").float()
        if var_t.ndim == 1:
            var_t = var_t.unsqueeze(-1)

        if _state.nlev is None:
            _state.nlev = int(var_t.shape[1])

        # Pad up to the full local node count (including halos/padding)
        # so downstream tensor shapes always match graph.n_nodes.
        full = torch.zeros(_graph.n_nodes, var_t.shape[1], device="cuda")
        full[: var_t.shape[0]] = var_t
        full_d = full.double()

        if _state.accum_sum is None:
            _state.accum_sum = torch.zeros_like(full_d)
            _state.accum_sumsq = torch.zeros_like(full_d)

        _state.accum_sum += full_d
        _state.accum_sumsq += full_d**2

        _state.accum_count += 1
        _state.current_step += 1

        comin.print_info(
            f"[rank={rank}] Dry run step {current_step + 1}/{dry_run_steps}"
        )
        return

    mean = (_state.accum_sum / _state.accum_count).float()
    e_x2 = (_state.accum_sumsq / _state.accum_count).float()
    var = torch.clamp(e_x2 - mean**2, min=1e-6)
    std = torch.sqrt(var)

    _state.accum_mean = mean  # (n_nodes, nlev) float32 CUDA tensor
    _state.accum_std = std
    _state.dryrun_done = True

    comin.print_info(
        f"[rank={rank}] Dry run complete after {_state.accum_count} steps: "
        f"mean range=[{float(mean.min())}, {float(mean.max())}], "
        f"std range=[{float(std.min())}, {float(std.max())}]"
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online rollout-based GNN training callback, called by ICON at each
    time step. Every `ROLLOUT_STEPS` steps we run one autoregressive
    rollout-training update (see gnn_online.OnlineGNNTrainer.train_rollout_step)."""

    if not has_gpu:
        return
    if not _state.dryrun_done:
        comin.print_info(f"[rank={rank}] Dry run not complete, skipping training")
        return

    current_step = _state.current_step
    _state.current_step += 1

    if (
        _state.trainer is not None
        and _state.step_len_seconds is not None
        and current_step > 0
    ):
        steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
        if current_step % steps_per_save == 0:
            save_checkpoint(_state.trainer, CHECKPOINT_PATH, compute_rank, current_step)
            comin.print_info(f"[rank={rank}] Checkpoint saved at step={current_step}")

    icon_var_nodes = extract_icon_cells(_state.icon_var, domain.cells.ncells)
    var_t = torch.as_tensor(icon_var_nodes, device="cuda").float()
    if var_t.ndim == 1:
        var_t = var_t.unsqueeze(-1)

    full = torch.zeros(_graph.n_nodes, var_t.shape[1], device="cuda")
    full[: var_t.shape[0]] = var_t

    mean = _state.accum_mean
    std = _state.accum_std
    var_norm = (full - mean) / std

    unix_seconds = _icon_time_unix_seconds()
    trainer = _get_trainer(nlev=_state.nlev)
    snapshot = trainer.prepare_snapshot(var_norm, unix_seconds)

    if _state.rollout_source is None:
        # First snapshot in a new rollout window: this is the source state.
        _state.rollout_source = snapshot
        _state.rollout_targets.clear()
    else:
        _state.rollout_targets.append(snapshot)
        if len(_state.rollout_targets) == ROLLOUT_STEPS:
            result = trainer.train_rollout_step(
                _state.rollout_source, list(_state.rollout_targets)
            )
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
            # Slide the rollout window forward: the most recent snapshot
            # becomes the source of the next window.
            _state.rollout_source = snapshot
            _state.rollout_targets.clear()

    # Run 1-step-ahead inference and write the prediction back to ICON —
    # same grid, no reverse YAC coupling needed.
    pred = trainer.predict(snapshot, n_steps=1)  # (n_nodes, nlev) normalized
    pred_denorm = pred * std + mean
    pred_valid = pred_denorm[: domain.cells.ncells].to(torch.float64)
    insert_icon_cells(pred_valid, _state.AI_var)


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """Cleanly tear down PyTorch distributed before MPI_Finalize."""
    if has_gpu and dist.is_initialized():
        dist.destroy_process_group()
        comin.print_info(f"[rank={rank}] PyTorch distributed destroyed")
