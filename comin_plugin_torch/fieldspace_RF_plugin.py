"""ComIn plugin: online RECONSTRUCTOR + FORECASTOR (fieldspace_RF_online.py)
inside a running ICON simulation on its native grid, e.g. R2B8.

Every ICON work rank owns a GPU and one model patch: complete R2B4 backbone
cells with all their descendants (icon_nested_mgrid.py). Once per
``MESSE_AE_SAMPLE_SECONDS`` of model time, the owned cells of the ICON
variable are moved to the patches, the first ``MESSE_AE_DRY_RUN_SECONDS``
only collect the per-cell mean/std, and after that every sample step trains
both models and forecasts the next sample step.

``var_predict`` at model time t holds the forecast valid at t, made one sample
step earlier: the field at t - lead plus the predicted increment. The log
scores it against persistence (``fc_rmse``, ``persistence``, in the variable's
units) and prints the per-level losses of both models.
"""

import os
import sys
import time

import comin
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

try:
    _PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # COMIN may exec the plugin without __file__
    _PLUGIN_DIR = os.environ.get("MESSE_PLUGIN_DIR", os.getcwd())
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PLUGIN_DIR))
sys.path[:0] = [p for p in (_PLUGIN_DIR, _PROJECT_ROOT) if p not in sys.path]

from fieldspace_RF_online import OnlineReconstructorForecaster
from icon_nested_mgrid import CellExchange, load_global_mgrid, patch_mgrid
from MEssE.utils.icon_online_helper import (
    RunningMeanStd,
    extract_icon_cells,
    insert_icon_cells,
    parse_icon_datetime,
    sample_interval,
    save_checkpoint,
    setup_mpi_dist,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
env = os.environ.get
DOMAIN_ID = 1
VARIABLE = env("MESSE_ICON_VAR", "uas")  # AES 10 m zonal wind (u_10m in NWP)
LEVELS = [int(v) for v in env("MESSE_AE_LEVELS", "4,6,7,8").split(",")]  # backbone first, last = ICON's grid
LATENT_LEVEL = int(env("MESSE_AE_LATENT_LEVEL", "6"))
GRID_DIR = env("MESSE_FS_GRID_DIR", "/pool/data/ICON/grids/mpim")
SAMPLE_SECONDS = float(env("MESSE_AE_SAMPLE_SECONDS", "600"))  # forecast lead and history spacing
DRY_RUN_SECONDS = float(env("MESSE_AE_DRY_RUN_SECONDS", "86400"))
SAVE_INTERVAL_SECONDS = float(env("MESSE_AE_SAVE_INTERVAL_SECONDS", "86400"))
MODEL = dict(
    n_history=int(env("MESSE_AE_N_HISTORY", "4")),
    latent_channels=int(env("MESSE_AE_LATENT_CHANNELS", "4")),
    n_blocks=int(env("MESSE_AE_N_BLOCKS", "2")),  # attention blocks per compression stage
    n_forecaster_blocks=int(env("MESSE_AE_N_FORECASTER_BLOCKS", "2")),
    att_dim=int(env("MESSE_AE_ATT_DIM", "128")),
    ema_decay=float(env("MESSE_AE_EMA_DECAY", "0.99")),  # frozen = EMA of the reconstructor
    lr_recon=float(env("MESSE_AE_LR_RECON", "2e-4")),
    lr_forecast=float(env("MESSE_AE_LR_FORECAST", "2e-4")),
)
CHECKPOINT_PATH = os.path.join(os.getcwd(), "saved_models", "fieldspace_rf_online.pt")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
DEVICE = torch.device("cuda", 0)

# ----------------------------------------------------------------------------
# Ranks, grids, patches
# ----------------------------------------------------------------------------
glob = comin.descrdata_get_global()
DEVICE_FLAG = comin.COMIN_FLAG_DEVICE if glob.has_device else 0
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
compute_comm, rank, n_ranks, has_gpu = setup_mpi_dist(comm)
# The cell exchange needs every work rank; the runscript puts 4 per node.
assert comm.allreduce(int(has_gpu)) == comm.Get_size(), "every ICON work rank needs its own GPU"

domain = comin.descrdata_get_domain(DOMAIN_ID)
global_mgrid = load_global_mgrid(GRID_DIR, LEVELS[0], LEVELS[-1])
assert len(global_mgrid[-1]["coords"]) == domain.cells.ncells_global, "finest level must be ICON's grid"
REFINEMENT = 4 ** (LEVELS[-1] - LEVELS[0])  # fine cells per backbone cell
N_BACKBONE = len(global_mgrid[0]["coords"])

# Local flat cell id c = (blk-1)*nproma + (idx-1), Fortran order as in
# extract_icon_cells; glb_index is already 1D.
_n_local = domain.cells.ncells
_owned_local = np.flatnonzero(np.ravel(np.asarray(domain.cells.decomp_domain), order="F")[:_n_local] == 0)
_owned_glb = np.asarray(domain.cells.glb_index, dtype=np.int64)[_owned_local] - 1

# Each backbone cell goes to the rank that owns most of its fine cells.
_counts = np.empty((n_ranks, N_BACKBONE), dtype=np.int64)
compute_comm.Allgather(np.bincount(_owned_glb // REFINEMENT, minlength=N_BACKBONE), _counts)
_owner = _counts.argmax(axis=0)
_backbone = np.flatnonzero(_owner == rank)
exchange = CellExchange.build(compute_comm, _owned_local, _owned_glb, _owner, _backbone, REFINEMENT)
mgrids = patch_mgrid(global_mgrid, torch.from_numpy(_backbone))
del global_mgrid
_per_rank = np.bincount(_owner, minlength=n_ranks)
comin.print_info(
    f"patches: R2B{LEVELS[0]} backbone cells per rank {_per_rank.min()}-{_per_rank.max()}, "
    f"R2B{LEVELS[-1]} cells {_per_rank.min() * REFINEMENT}-{_per_rank.max() * REFINEMENT}"
)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
class State:
    icon_var = None
    predict_var = None
    first_seconds = None  # model time of the first callback
    sample_seconds = None  # SAMPLE_SECONDS rounded to whole ICON steps
    normalizer = None
    trainer = None
    step = 0  # sample steps since the dry run
    prev_patch = None  # physical patch one sample step ago
    forecast = None  # physical forecast valid at the next sample step


def _seconds() -> float:
    return parse_icon_datetime(comin.current_get_datetime()).timestamp()


def _patch() -> torch.Tensor:
    """Collective: this rank's patch of the ICON variable, (n_fine, nlev) on the GPU."""
    cells = extract_icon_cells(State.icon_var, domain.cells.ncells)
    cells = cells.get() if hasattr(cells, "get") else cells  # cupy -> numpy for MPI
    return torch.from_numpy(exchange.to_patch(compute_comm, cells.reshape(len(cells), -1))).to(DEVICE)


def _write(forecast: torch.Tensor) -> None:
    """Collective: the forecast's owned cells into ``var_predict``."""
    owned = exchange.from_patch(compute_comm, forecast.cpu().numpy())
    insert_icon_cells(owned.astype(np.float64), State.predict_var, indices=exchange.send_ids)


def _trainer(nlev: int) -> OnlineReconstructorForecaster:
    trainer = OnlineReconstructorForecaster(mgrids, LEVELS, LATENT_LEVEL, nlev, device=DEVICE, **MODEL)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        State.step = ckpt["step"]
        comin.print_info(f"restored {CHECKPOINT_PATH} at step {State.step}")
    n_recon = sum(p.numel() for p in trainer.model.reconstructor.parameters())
    n_forecast = sum(p.numel() for p in trainer.model.forecaster.parameters())
    comin.print_info(f"reconstructor {n_recon:,} / forecaster {n_forecast:,} parameters, {MODEL}")
    return trainer


def _levels(values: dict) -> str:
    return " ".join(f"{'R2B' if z == 0 else 'r'}{LEVELS[0] + z}={v:.3f}" for z, v in values.items())


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
comin.var_request_add(("var_predict", DOMAIN_ID), lmodexclusive=True)
comin.metadata_set(("var_predict", DOMAIN_ID), zaxis_id=comin.COMIN_ZAXIS_2D)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    ep = [comin.EP_ATM_WRITE_OUTPUT_BEFORE]
    State.icon_var = comin.var_get(ep, (VARIABLE, DOMAIN_ID), comin.COMIN_FLAG_READ | DEVICE_FLAG)
    State.predict_var = comin.var_get(ep, ("var_predict", DOMAIN_ID), comin.COMIN_FLAG_WRITE | DEVICE_FLAG)


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def sample_step():
    """Acts once per sample interval of model time. All ranks see the same
    ICON time, so they take the same branches, as the exchange and DDP need."""
    now = _seconds()
    if State.first_seconds is None:
        State.first_seconds = now
        stride, State.sample_seconds = sample_interval(SAMPLE_SECONDS, comin.descrdata_get_timesteplength(DOMAIN_ID))
        comin.print_info(f"sampling every {stride} ICON steps = {State.sample_seconds} s")
        n_dry = int(DRY_RUN_SECONDS // State.sample_seconds)
        comin.print_info(f"dry run: {n_dry} sample steps = {n_dry * stride} ICON steps = {n_dry * State.sample_seconds:.0f} s")
    if round(now - State.first_seconds) % State.sample_seconds:
        return
    patch = _patch()

    if State.normalizer is None:
        State.normalizer = RunningMeanStd(patch.shape, int(DRY_RUN_SECONDS // State.sample_seconds), DEVICE)
    if not State.normalizer.update(patch):
        return
    if State.trainer is None:
        State.trainer = _trainer(nlev=patch.shape[1])
    std = State.normalizer.stats.std

    # Score and write the forecast made one sample step ago, which is valid now.
    scores = ""
    if State.forecast is not None:
        fc_rmse = (State.forecast - patch).square().mean().sqrt()
        persistence = (State.prev_patch - patch).square().mean().sqrt()
        scores = f"fc_rmse={fc_rmse:.4f} persistence={persistence:.4f} "
        _write(State.forecast)

    torch.cuda.synchronize()
    start = time.perf_counter()
    stats = State.trainer.train_step(State.normalizer.normalize(patch), now)
    increment = State.trainer.predict_increment(now + State.sample_seconds)
    torch.cuda.synchronize()
    State.forecast = None if increment is None else patch + increment * std
    State.prev_patch = patch

    if stats:
        comin.print_info(
            f"step={State.step} {scores}"
            f"recon[{_levels(stats['recon'])}] incre[{_levels(stats['incre'])}] "
            f"oracle[{_levels(stats['oracle'])}] grad_norm[recon={stats['grad_norm']['recon']:.2f} "
            f"forecast={stats['grad_norm']['forecast']:.2f}] train_s={time.perf_counter() - start:.2f} "
            f"torch_peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.1f}"
        )
    State.step += 1
    if State.step % max(1, int(SAVE_INTERVAL_SECONDS // State.sample_seconds)) == 0:
        norm = State.normalizer.stats
        save_checkpoint(State.trainer, CHECKPOINT_PATH, rank, State.step, {"norm_mean": norm.mean, "norm_std": norm.std})


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    if dist.is_initialized():
        dist.destroy_process_group()
