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

from graph_utils import build_local_graph
from gnn_online import GNNSnapshot, OnlineGNNTrainer
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
ROLLOUT_STEPS = int(os.environ.get("MESSE_GNN_ROLLOUT_STEPS", "4"))  # forecast horizon, in steps
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
# MPI and PyTorch distributed setup (DDP)
# ----------------------------------------------------------------------------

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
size = comm.Get_size()
rank = comm.Get_rank()

# GPU ranks and DDP world
compute_comm, compute_rank, compute_size, has_gpu = setup_mpi_dist(comm)
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)


# ----------------------------------------------------------------------------
# Native ICON grid + local patch graph (no YAC, no HEALPix)
# ----------------------------------------------------------------------------
domain = comin.descrdata_get_domain(DOMAIN_ID)

_graph = None  # built lazily in sec_ctor / setup_graph, once nproma is known
_owned_idx_np: Optional[np.ndarray] = None  # flat cell ids of owned (non-halo) nodes
if has_gpu:
    _graph = build_local_graph(
        domain, nproma=glob.nproma, device=torch.device("cuda", 0)
    )
    # NOTE: owned_cells + HALO cells are presented in local graph, but only owned cells are valid for writing back to ICON.
    _owned_idx_np = np.nonzero(_graph.owned_mask.cpu().numpy())[0]
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
        # dry run for normalization (see online_training.RunningMeanStd)
        "normalizer",
        # source snapshot + fixed-horizon scheduling (ROLLOUT_STEPS steps
        # ahead); see online_training.RolloutBuffer
        "rollout",
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


def _extract_padded(comin_var) -> torch.Tensor:
    """Extract this rank's ICON cells for `comin_var` and zero-pad to the
    full local graph node count (`_graph.n_nodes`, includes halos/unused
    padding), so downstream tensor shapes always match the graph regardless
    of how many cells ICON actually returned.
    """
    cells = extract_icon_cells(comin_var, domain.cells.ncells)  # (ncells, nlev) or (ncells,)
    t = torch.as_tensor(xp.asarray(cells), device="cuda").float()
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    full = torch.zeros(_graph.n_nodes, t.shape[1], device="cuda")
    full[: t.shape[0]] = t
    return full


def _maybe_save_checkpoint(current_step: int) -> None:
    if _state.trainer is None or _state.step_len_seconds is None or current_step == 0:
        return
    steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
    if current_step % steps_per_save == 0:
        save_checkpoint(_state.trainer, CHECKPOINT_PATH, compute_rank, current_step)
        comin.print_info(f"[rank={rank}] Checkpoint saved at step={current_step}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def dry_run():
    """Dry run to estimate per-node mean/std over the first
    `DRY_RUN_TIME_SECONDS` seconds, directly on the ICON native grid
    (no HEALPix reshaping needed). See online_training.RunningMeanStd."""

    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(1))

    if not has_gpu:
        return
    if _state.normalizer is not None and _state.normalizer.done:
        return

    dry_run_steps = DRY_RUN_TIME_SECONDS // _state.step_len_seconds
    full = _extract_padded(_state.icon_var)

    if _state.nlev is None:
        _state.nlev = int(full.shape[1])
    if _state.normalizer is None:
        _state.normalizer = RunningMeanStd(
            shape=full.shape, n_samples=dry_run_steps, device=torch.device("cuda", 0)
        )

    step = _state.current_step
    done = _state.normalizer.update(full)
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
    """Online rollout-based GNN training callback, called by ICON at each
    time step. Every `ROLLOUT_STEPS` steps we run one training update:
    the model rolls forward `ROLLOUT_STEPS` autoregressive steps from a
    source snapshot, and the loss is evaluated once against the single
    real snapshot observed that many steps later (see
    gnn_online.OnlineGNNTrainer.train_rollout_step and
    online_training.RolloutBuffer for the fixed-horizon scheduling)."""

    if not has_gpu:
        return
    if _state.normalizer is None or not _state.normalizer.done:
        comin.print_info(f"[rank={rank}] Dry run not complete, skipping training")
        return

    current_step = _state.current_step
    _state.current_step += 1
    _maybe_save_checkpoint(current_step)

    full = _extract_padded(_state.icon_var)
    var_norm = _state.normalizer.normalize(full)

    trainer = _get_trainer(nlev=_state.nlev)
    snapshot = trainer.prepare_snapshot(var_norm, _icon_time_unix_seconds())

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
    pred = trainer.predict(snapshot, n_steps=1)  # (n_nodes, nlev) normalized
    pred_denorm = _state.normalizer.denormalize(pred)
    # Only write back predictions for owned (non-halo) cells: halo nodes are
    # only present in the graph so message passing near the patch boundary
    # sees correct neighbor information (see graph_utils.py).
    pred_owned_np = pred_denorm[_owned_idx_np].cpu().numpy()
    insert_icon_cells(
        pred_owned_np.astype(np.float64), _state.AI_var, indices=_owned_idx_np
    )


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """Cleanly tear down PyTorch distributed before MPI_Finalize."""
    if has_gpu and dist.is_initialized():
        dist.destroy_process_group()
        comin.print_info(f"[rank={rank}] PyTorch distributed destroyed")
