"""ComIn plugin: online training of the multi-level Field-Space autoencoder
forecaster (fieldspace_AE_online.py) inside a running ICON simulation.

Every ICON work rank must own a GPU. At startup each rank reads the grid files
of all levels from the backbone to ICON's own grid, the backbone cells are
assigned to ranks, and each rank builds its model patch and the cell exchange
(icon_nested_mgrid.py). Every step, all ranks move ICON's owned cell values to
the patches, train and predict on their patch, and move the predictions back
for write-back into ``var_predict``.
"""

import comin
import os
import sys
import time
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

from icon_nested_mgrid import (
    MIN_BACKBONE_CELLS,
    CellExchange,
    assign_backbone_owners,
    backbone_counts,
    build_nested_mgrid,
    check_cell_centers,
    load_level_grids,
    mpim_grid_paths,
)
from MEssE.comin_plugin_torch.fieldspace_AE_online import OnlineFieldSpaceAETrainer
from MEssE.utils.icon_online_helper import (
    RunningMeanStd,
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
# "uas" in AES physics; the NWP physics name for the same field is "u_10m".
ICON_VARIABLE_NAME = os.environ.get("MESSE_ICON_VAR", "uas")
# R2B levels: backbone first, then residual levels; the finest must be ICON's grid.
MODEL_LEVELS = sorted(int(v) for v in os.environ.get("MESSE_AE_LEVELS", "4,6,7,8").split(","))
# Residual levels finer than this are folded into it (see fieldspace_AE_online).
LATENT_LEVEL = int(os.environ.get("MESSE_AE_LATENT_LEVEL", "6"))
# Directory with the MPI-M Earth_IcosS_* grid file of every level from backbone to fine.
GRID_DIR = os.environ.get("MESSE_FS_GRID_DIR", "/pool/data/ICON/grids/mpim")
N_HISTORY = int(os.environ.get("MESSE_AE_N_HISTORY", "4"))  # cached latents per prediction
ROLLOUT_STEPS = int(os.environ.get("MESSE_AE_ROLLOUT_STEPS", "1"))  # training horizon, in steps
LATENT_CHANNELS = int(os.environ.get("MESSE_AE_LATENT_CHANNELS", "4"))  # channels per latent-level cell
N_BLOCKS = int(os.environ.get("MESSE_AE_N_BLOCKS", "1"))  # attention blocks per encoder/decoder stage
N_PROCESSOR_BLOCKS = int(os.environ.get("MESSE_AE_N_PROCESSOR_BLOCKS", "2"))
RECON_WEIGHT = float(os.environ.get("MESSE_AE_RECON_WEIGHT", "1.0"))
ATT_DIM = int(os.environ.get("MESSE_AE_ATT_DIM", "32"))
N_HEAD_CHANNELS = int(os.environ.get("MESSE_AE_N_HEAD_CHANNELS", "8"))
HIDDEN_DIM = int(os.environ.get("MESSE_AE_HIDDEN_DIM", "64"))  # compression MLP width

EXPERIMENTS_DIR = os.path.abspath(os.getcwd())
SAVED_MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "saved_models")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
# Not fieldspace_online.pt: the two models' state dicts are incompatible.
CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "fieldspace_ae_online.pt")
DRY_RUN_TIME_SECONDS = int(os.environ.get("MESSE_AE_DRY_RUN_SECONDS", "3600"))  # 1 hour
SAVE_INTERVAL_SECONDS = int(os.environ.get("MESSE_AE_SAVE_INTERVAL_SECONDS", "86400"))  # 1 day

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

compute_comm, compute_rank, compute_size, has_gpu = setup_mpi_dist(comm)
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)
# The plugin runs on every ICON work rank, and the cell exchange needs all of
# them: a work rank without its own GPU (SLURM_LOCALID >= 4) means a wrong task
# layout in the runscript. The count is the same on every rank, so all raise.
if num_calculate_processes != size:
    raise RuntimeError(
        f"only {num_calculate_processes} of {size} ICON work ranks have their own GPU; "
        "every work rank needs one (check the runscript's MPI rank layout)"
    )


# ----------------------------------------------------------------------------
# Grids, backbone ownership, model patch and cell exchange (no YAC, no HEALPix)
# ----------------------------------------------------------------------------
domain = comin.descrdata_get_domain(DOMAIN_ID)
BASE_LEVEL, FINE_LEVEL = MODEL_LEVELS[0], MODEL_LEVELS[-1]
REFINEMENT = 4 ** (FINE_LEVEL - BASE_LEVEL)  # fine cells per backbone cell


def _setup_patch():
    grids = load_level_grids(
        mpim_grid_paths(GRID_DIR, BASE_LEVEL, FINE_LEVEL), ncells_global=domain.cells.ncells_global
    )
    n_local = int(domain.cells.ncells)
    # Flat local cell id c = (blk-1)*nproma + (idx-1), Fortran order, as in
    # extract_icon_cells/insert_icon_cells. glb_index is already 1D.
    decomp = np.ravel(np.asarray(domain.cells.decomp_domain), order="F")[:n_local]
    glb = np.ravel(np.asarray(domain.cells.glb_index)).astype(np.int64)[:n_local] - 1
    owned_local = np.flatnonzero(decomp == 0)
    owned_glb = glb[owned_local]
    clon = np.ravel(np.asarray(domain.cells.clon), order="F")[:n_local]
    clat = np.ravel(np.asarray(domain.cells.clat), order="F")[:n_local]
    check_cell_centers(grids[FINE_LEVEL], owned_glb, clon[owned_local], clat[owned_local])

    n_backbone_global = grids[BASE_LEVEL].n_cells
    counts = np.empty((compute_size, n_backbone_global), dtype=np.int64)
    compute_comm.Allgather(backbone_counts(owned_glb, n_backbone_global, REFINEMENT), counts)
    owner = assign_backbone_owners(counts, REFINEMENT)  # same result, or same error, on every rank
    backbone_per_rank = np.bincount(owner, minlength=compute_size)
    if backbone_per_rank.min() < MIN_BACKBONE_CELLS:
        raise ValueError(
            f"a rank got only {backbone_per_rank.min()} R2B{BASE_LEVEL} backbone cells, need "
            f"{MIN_BACKBONE_CELLS}: use a coarser backbone level or fewer GPUs"
        )
    my_backbone = np.flatnonzero(owner == compute_rank)
    exchange = CellExchange.build(compute_comm, owned_local, owned_glb, owner, my_backbone, REFINEMENT)
    mgrid = build_nested_mgrid(grids, my_backbone, device=torch.device("cuda", 0))

    owned_per_rank = counts.sum(axis=1)
    comin.print_info(
        f"model patches: levels={MODEL_LEVELS}, latent_level={LATENT_LEVEL}; per rank: "
        f"R2B{BASE_LEVEL} backbone cells {backbone_per_rank.min()}-{backbone_per_rank.max()}, "
        f"R2B{FINE_LEVEL} patch cells {backbone_per_rank.min() * REFINEMENT}-"
        f"{backbone_per_rank.max() * REFINEMENT}, ICON-owned cells "
        f"{owned_per_rank.min()}-{owned_per_rank.max()}; rank {rank} keeps "
        f"{int(exchange.send_counts[compute_rank])} of its {owned_glb.size} owned cells"
    )
    return mgrid, exchange


_mgrid, _exchange = _setup_patch()


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
        "patch",
        "patch_time",
        # forecast waiting for its valid time before it goes to var_predict
        "forecast",
        "forecast_seconds",
        # dry run for normalization (see icon_online_helper.RunningMeanStd)
        "normalizer",
        # model + latent reservoir (see fieldspace_AE_online.OnlineFieldSpaceAETrainer)
        "trainer",
    )

    def __init__(self) -> None:
        self.current_step: int = 0
        self.nlev: Optional[int] = None
        self.step_len_seconds: Optional[int] = None

        self.icon_var = None
        self.AI_var = None
        self.patch: Optional[torch.Tensor] = None
        self.patch_time: Optional[str] = None
        self.forecast: Optional[np.ndarray] = None  # (n_fine, nlev) denormalized, patch order
        self.forecast_seconds: Optional[float] = None  # its valid time, unix seconds

        self.normalizer: Optional[RunningMeanStd] = None
        self.trainer: Optional[OnlineFieldSpaceAETrainer] = None


_state = _State()


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def _icon_time_unix_seconds() -> float:
    return float(parse_icon_datetime(comin.current_get_datetime()).timestamp())


def _current_patch() -> torch.Tensor:
    """Collective: this rank's model patch of the ICON variable at the
    current model time, (n_fine, nlev) float32 on the GPU in nested patch
    order. Exchanged once per model time and reused by both callbacks."""
    now = comin.current_get_datetime()
    if _state.patch_time != now:
        cells = extract_icon_cells(_state.icon_var, domain.cells.ncells)  # (ncells, nlev) or (ncells,)
        host = cells.get() if hasattr(cells, "get") else np.asarray(cells)
        if host.ndim == 1:
            host = host[:, None]
        patch = _exchange.to_patch(compute_comm, host)
        _state.patch = torch.from_numpy(patch).to(torch.device("cuda", 0))
        _state.patch_time = now
    return _state.patch


def _get_trainer(nlev: int) -> OnlineFieldSpaceAETrainer:
    if _state.trainer is not None:
        return _state.trainer

    _state.trainer = OnlineFieldSpaceAETrainer(
        nlev=nlev,
        mgrid=_mgrid,
        levels=MODEL_LEVELS,
        latent_level=LATENT_LEVEL,
        n_history=N_HISTORY,
        rollout_steps=ROLLOUT_STEPS,
        latent_channels=LATENT_CHANNELS,
        n_blocks=N_BLOCKS,
        n_processor_blocks=N_PROCESSOR_BLOCKS,
        recon_weight=RECON_WEIGHT,
        lr=float(os.environ.get("MESSE_AE_LR", "2e-4")),
        att_dim=ATT_DIM,
        n_head_channels=N_HEAD_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        grad_clip=1.0,
        use_ddp=dist.is_initialized(),
        device=torch.device("cuda", 0),
        log_fn=comin.print_info,
        rank=rank,
    )

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=torch.device("cuda", 0), weights_only=False)
        _state.trainer.model.load_state_dict(ckpt["model_state_dict"])
        _state.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        saved_step = ckpt.get("step", "unknown")
        _state.current_step = int(saved_step) if isinstance(saved_step, int) else 0
        comin.print_info(
            f"[rank={rank}] Restored FieldSpace AE model from checkpoint {CHECKPOINT_PATH} "
            f"(step={saved_step}), resuming from step={_state.current_step}; the latent "
            f"reservoir is not checkpointed and refills over the next "
            f"{_state.trainer.reservoir.capacity} steps"
        )
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


def _maybe_save_checkpoint(current_step: int) -> None:
    if _state.trainer is None:
        return
    if _state.step_len_seconds is None or current_step == 0:
        return
    steps_per_save = max(1, SAVE_INTERVAL_SECONDS // _state.step_len_seconds)
    if current_step % steps_per_save == 0:
        save_checkpoint(_state.trainer, CHECKPOINT_PATH, compute_rank, current_step)
        comin.print_info(f"[rank={rank}] Checkpoint saved at step={current_step}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def dry_run():
    """Dry run to estimate per-cell mean/std of the patch over the first
    `DRY_RUN_TIME_SECONDS` seconds. See icon_online_helper.RunningMeanStd."""

    _state.step_len_seconds = int(comin.descrdata_get_timesteplength(DOMAIN_ID))
    if _state.normalizer is not None and _state.normalizer.done:
        return

    dry_run_steps = max(1, DRY_RUN_TIME_SECONDS // _state.step_len_seconds)
    patch = _current_patch()

    if _state.nlev is None:
        _state.nlev = int(patch.shape[1])
    if _state.normalizer is None:
        _state.normalizer = RunningMeanStd(
            shape=patch.shape, n_samples=dry_run_steps, device=torch.device("cuda", 0)
        )

    step = _state.current_step
    done = _state.normalizer.update(patch)
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
    """Online training callback, called by ICON at each time step.

    Every step, the current field is compressed and its latent cached, with
    its ICON model time, in the trainer's reservoir. Once the reservoir holds
    `N_HISTORY + ROLLOUT_STEPS - 1` earlier latents, every step also trains
    (see fieldspace_AE_online.OnlineFieldSpaceAETrainer.train_step).

    `var_predict` at model time t holds the forecast *valid at t*, made one
    step earlier from data up to t - dt (the forecast the loss scores at t),
    so it can be compared directly with the ICON variable in the same output
    snapshot. Each step's new 1-step-ahead forecast is kept until the next
    step, then sent to the rank owning each cell and written. Every rank takes
    the same branches on the same steps, as the collective exchange and DDP
    require.
    """

    if _state.normalizer is None or not _state.normalizer.done:
        comin.print_info(f"[rank={rank}] Dry run not complete, skipping training")
        return

    current_step = _state.current_step
    _state.current_step += 1
    _maybe_save_checkpoint(current_step)

    patch_norm = _state.normalizer.normalize(_current_patch())

    trainer = _get_trainer(nlev=_state.nlev)
    now_seconds = _icon_time_unix_seconds()
    snapshot = trainer.prepare_snapshot(patch_norm, now_seconds)

    # Collective: the forecast made last step for this model time goes to the
    # ranks owning its cells. Times are identical on all ranks.
    if _state.forecast is not None and _state.forecast_seconds == now_seconds:
        owned_forecast = _exchange.from_patch(compute_comm, _state.forecast)
        insert_icon_cells(owned_forecast.astype(np.float64), _state.AI_var, indices=_exchange.send_local_ids)
    _state.forecast = None

    torch.cuda.synchronize()
    train_seconds = time.perf_counter()
    result = trainer.train_step(snapshot)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_seconds
    # PyTorch's own peak, and what is left on the device next to ICON.
    gpu_free_bytes, _ = torch.cuda.mem_get_info()
    cost = (
        f"train_s={train_seconds:.2f} torch_peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.1f} "
        f"gpu_free_GiB={gpu_free_bytes / 2**30:.1f}"
    )
    reservoir = f"reservoir={len(trainer.reservoir)}/{trainer.reservoir.capacity}"
    if result.get("skipped", False) and not result.get("needs_rollback", False):
        # Reservoir warm-up, or a skipped input (the trainer logs why).
        comin.print_info(f"[rank={rank}] step={current_step} no update, {reservoir} {cost}")
    else:
        comin.print_info(
            f"[rank={rank}] step={current_step} loss={result['loss']:.6f} "
            f"pred_mse={result['loss_dict'].get('train/MSE_pred', float('nan')):.6f} "
            f"recon_mse={result['loss_dict'].get('train/MSE_recon', float('nan')):.6f} "
            f"grad_norm={result.get('grad_norm', 0.0):.4f} "
            f"skipped={result.get('skipped', False)} {reservoir} {cost}"
        )
    if result.get("needs_rollback"):
        comin.print_info(
            f"[rank={rank}] step={current_step} NaN detected — "
            "rolling back to last checkpoint"
        )
        # train_step has already cleared the reservoir: its latents came from
        # the model being discarded.
        rollback_checkpoint(_state.trainer, CHECKPOINT_PATH)

    pred = trainer.predict(n_steps=1)  # (n_fine, nlev) normalized, or None
    if pred is not None:
        # Written to var_predict next step, when its valid time is reached.
        _state.forecast = _state.normalizer.denormalize(pred).cpu().numpy()
        _state.forecast_seconds = trainer.reservoir.latest_seconds + trainer.step_seconds


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    """Cleanly tear down PyTorch distributed before MPI_Finalize."""
    if dist.is_initialized():
        dist.destroy_process_group()
        comin.print_info(f"[rank={rank}] PyTorch distributed destroyed")
