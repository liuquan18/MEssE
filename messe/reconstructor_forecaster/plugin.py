"""ComIn plugin: training RECONSTRUCTOR + FORECASTER (RF) in real time while ICON
is running. 

ICON in R2B8 is running on 32 GPUs, and the AI model is trained on the same
devices with DDP. The data on each GPU at each time step is accessed through 
ComIn. 

ICON is firstly dry run for ``run.dry_run_seconds`` of model time, collecting 
the per-cell mean/std for normalization. After that, every ``run.sample_seconds`` 
of model time, the RF is trained. 

The forcing fields are `forcings.icon_fields`` (SST, sea ice), ``forcings.insolation``
(TOA incoming shortwave flux), ``forcings.statics`` (orography, ``oromea``), 
``forcings.position`` (cell centre as a unit vector on the sphere), and 
``forcings.orientation`` (the direction of the backbone cell (up or down)).

"""

import os
import shutil
import time

import comin
import numpy as np
import torch
import xarray as xr
import torch.distributed as dist
from fieldspacenn.src.data.normalizer import MeanStdNormalizer
from mpi4py import MPI

from messe.common_utils.icon_data import (
    extract_icon_cells,
    insert_icon_cells,
    parse_icon_datetime,
    sample_interval,
    setup_mpi_dist,
)
from messe.common_utils.nested_grid import CellExchange, ghost_cells, load_global_mgrid, patch_mgrid
from messe.common_utils.training import RunningMeanStd, mean_stats, save_checkpoint
from messe.reconstructor_forecaster.config import load_config
from messe.reconstructor_forecaster.forcings import cell_position, toa_insolation
from messe.reconstructor_forecaster.model import OnlineReconstructorForecaster

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CONFIG_PATH = os.environ["MESSE_RF_CONFIG"]
CONFIG = load_config(CONFIG_PATH)
DOMAIN_ID = 1
LEVELS = CONFIG.grid.levels
FORCING_NAMES = CONFIG.forcings.names
CHECKPOINT_PATH = os.path.join(os.getcwd(), "saved_models", "fieldspace_rf_online.pt")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
DEVICE = torch.device("cuda", 0)

# ----------------------------------------------------------------------------
# Ranks, grids, patches
# ----------------------------------------------------------------------------
comm =MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
compute_comm, rank, n_ranks, has_gpu = setup_mpi_dist(comm)
comin.print_info(f"config {CONFIG_PATH}: {CONFIG}")
if rank == 0:
    shutil.copy(CONFIG_PATH, os.path.join(os.getcwd(), "rf_config_used.yaml"))
# The cell exchange needs every work rank; the runscript puts 4 per node.
assert comm.allreduce(int(has_gpu)) == comm.Get_size(), "every ICON work rank needs its own GPU"

domain = comin.descrdata_get_domain(DOMAIN_ID)
global_mgrid = load_global_mgrid(CONFIG.grid.dir, LEVELS[0], LEVELS[-1], canonical=CONFIG.grid.canonical_order)
assert len(global_mgrid[-1]["coords"]) == domain.cells.ncells_global, "finest level must be ICON's grid"
REFINEMENT = 4 ** (LEVELS[-1] - LEVELS[0])  # fine cells per backbone cell
N_BACKBONE = len(global_mgrid[0]["coords"])
# The grids' cell order: ICON's, or the canonical one (same backbone cell, another place in it).
_icon_id = global_mgrid[-1].get("file_index", torch.arange(domain.cells.ncells_global)).numpy()  # of every grid cell
_grid_id = np.empty_like(_icon_id)  # of every ICON cell
_grid_id[_icon_id] = np.arange(len(_icon_id))

# Local flat cell id c = (blk-1)*nproma + (idx-1), Fortran order as in
# extract_icon_cells; glb_index is already 1D.
_n_local = domain.cells.ncells
_owned_local = np.flatnonzero(np.ravel(np.asarray(domain.cells.decomp_domain), order="F")[:_n_local] == 0)
_owned_glb = _grid_id[np.asarray(domain.cells.glb_index, dtype=np.int64)[_owned_local] - 1]

# Each backbone cell goes to the rank that owns most of its fine cells.
_counts = np.empty((n_ranks, N_BACKBONE), dtype=np.int64)
compute_comm.Allgather(np.bincount(_owned_glb // REFINEMENT, minlength=N_BACKBONE), _counts)
_owner = _counts.argmax(axis=0)
# The patch: this rank's backbone cells, then (grid.ghost_rings) those of other ranks that touch them.
_touching = global_mgrid[0]["adjc"].numpy() if CONFIG.grid.ghost_rings else None
_owned_backbone = np.flatnonzero(_owner == rank)
_ghosts = ghost_cells(_owned_backbone, _touching) if CONFIG.grid.ghost_rings else _owned_backbone[:0]
_backbone = np.concatenate([_owned_backbone, _ghosts])
N_OWNED = len(_owned_backbone) * REFINEMENT  # the patch's first fine cells; the rest are ghost cells
exchange = CellExchange.build(compute_comm, _owned_local, _owned_glb, _owner, _backbone, REFINEMENT, DEVICE, _touching)
mgrids = patch_mgrid(global_mgrid, torch.from_numpy(_backbone))
FINE_LON, FINE_LAT = mgrids[-1]["coords"].to(DEVICE).T  # patch cell centres [radian], for the insolation
# Static fields of the patch: its fine cells are the backbone cells' descendants, in nested order.
_patch_ids = _icon_id[(_backbone[:, None] * REFINEMENT + np.arange(REFINEMENT)).ravel()]
STATIC = np.empty((len(_patch_ids), len(CONFIG.forcings.statics)), dtype=np.float32)
with xr.open_dataset(CONFIG.forcings.static_file) as _ds:
    for _k, _name in enumerate(CONFIG.forcings.statics):
        STATIC[:, _k] = _ds[_name].values[_patch_ids]
STATIC = torch.from_numpy(STATIC).to(DEVICE)
if CONFIG.forcings.position:  # after the file's statics, as in CONFIG.forcings.names
    STATIC = torch.cat([STATIC, cell_position(FINE_LON, FINE_LAT).float()], dim=1)
if CONFIG.forcings.orientation:  # every fine cell gets its backbone cell's
    STATIC = torch.cat([STATIC, mgrids[0]["frame_direction"].to(DEVICE).repeat_interleave(REFINEMENT, dim=0)], dim=1)
del global_mgrid
_per_rank = np.bincount(_owner, minlength=n_ranks)
comin.print_info(
    f"patches: R2B{LEVELS[0]} backbone cells per rank {_per_rank.min()}-{_per_rank.max()}, "
    f"R2B{LEVELS[-1]} cells {_per_rank.min() * REFINEMENT}-{_per_rank.max() * REFINEMENT}; "
    f"rank 0 has {len(_ghosts)} ghost backbone cells"
)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
class State:
    icon_var = None
    predict_var = None
    forcing_vars = []
    forcing_norm = None  # MeanStdNormalizer, one global mean/std per forcing
    first_seconds = None  # model time of the first callback
    sample_seconds = None  # run.sample_seconds rounded to whole ICON steps
    normalizer = None
    trainer = None
    step = 0  # sample steps since the dry run
    forecast = None  # physical forecast valid at the next sample step
    log_rows = []  # per-step values since the last log line


def _seconds() -> float:
    return parse_icon_datetime(comin.current_get_datetime()).timestamp()


def _patch(variables) -> torch.Tensor:
    """Collective: this rank's patch of the ICON variables, (n_fine, levels
    of all variables), from ICON's GPU memory to the patches without leaving
    the GPU."""
    n = domain.cells.ncells
    columns = [torch.as_tensor(extract_icon_cells(var, n), device=DEVICE).reshape(n, -1).float() for var in variables]
    return exchange.to_patch(compute_comm, torch.cat([torch.empty(n, 0, device=DEVICE), *columns], dim=1))


def _global_mean_std(patch: torch.Tensor):
    """Collective: mean and std of every column over the owned cells of all
    patches, i.e. over every ICON cell once."""
    values = patch[:N_OWNED].double()
    sums = torch.stack([values.sum(dim=0), values.square().sum(dim=0)])
    torch.cuda.synchronize()  # MPI does not wait for the CUDA stream that wrote sums
    compute_comm.Allreduce(MPI.IN_PLACE, sums)
    mean, mean_square = sums / compute_comm.allreduce(len(values))
    return mean.float(), (mean_square - mean**2).clamp(min=1e-12).sqrt().float()


def _stats(names, mean, std) -> str:
    return " ".join(f"{name}={m:.4g}±{s:.4g}" for name, m, s in zip(names, mean.tolist(), std.tolist()))


def _write(forecast: torch.Tensor) -> None:
    """Collective: the forecast's owned cells into ``var_predict``."""
    owned = exchange.from_patch(compute_comm, forecast)
    insert_icon_cells(owned.double(), State.predict_var, indices=exchange.owned_ids)


def _forcing(seconds: float) -> torch.Tensor:
    """Collective: this rank's physical forcing patch at model time
    ``seconds``, (n_fine, len(FORCING_NAMES))."""
    columns = [_patch(State.forcing_vars)]
    if CONFIG.forcings.insolation:
        columns.append(toa_insolation(seconds, FINE_LON, FINE_LAT)[:, None])
    return torch.cat(columns + [STATIC], dim=1)


def _trainer(nlev: int, seconds: float) -> OnlineReconstructorForecaster:
    """Collective: normalizes the forcings by their global mean/std at
    ``seconds``, then builds the trainer."""
    mean, std = _global_mean_std(_forcing(seconds))
    comin.print_info(f"forcings (mean±std over all cells): {_stats(FORCING_NAMES, mean, std)}")
    flat = [name for name, m, s in zip(FORCING_NAMES, mean.tolist(), std.tolist()) if s <= 1e-6 * (1 + abs(m))]
    if flat:  # the same on every rank: the statistics are global
        raise RuntimeError(f"forcings without spread over the globe, read wrongly? {flat}")
    State.forcing_norm = MeanStdNormalizer({"mean": mean, "std": std}, {})
    trainer = OnlineReconstructorForecaster(
        mgrids,
        LEVELS,
        CONFIG.grid.latent_level,
        nlev,
        n_owned=len(_owned_backbone),
        device=DEVICE,
        **CONFIG.trainer_kwargs(State.sample_seconds),
    )
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        State.step = trainer.n_updates = ckpt["step"]
        comin.print_info(f"restored {CHECKPOINT_PATH} at step {State.step}")
    n_recon = sum(p.numel() for p in trainer.model.reconstructor.parameters())
    n_forecast = sum(p.numel() for p in trainer.model.forecaster.parameters())
    comin.print_info(f"reconstructor {n_recon:,} / forecaster {n_forecast:,} parameters")
    return trainer


def _name(zoom: int) -> str:
    return f"{'R2B' if zoom == 0 else 'r'}{LEVELS[0] + zoom}"


def _levels(values: dict) -> str:
    return " ".join(f"{_name(z)}={v:.4g}" for z, v in values.items())


def _log_line() -> str:
    """The means of ``State.log_rows``."""
    means = mean_stats(State.log_rows)
    losses = " ".join(f"{key}[{_levels(means[key])}]" for key in ("recon", "state_ms", "latent_mse", "latent_persist"))
    if "memory" in means:
        losses += " memory[" + " ".join(f"{key}={value:.4g}" for key, value in means["memory"].items()) + "]"
    return (
        f"step={State.step} fc_rmse={means['fc_rmse']:.4f} {losses} "
        f"train_s={means['train_s']:.2f} torch_peak_GiB={torch.cuda.max_memory_allocated() / 2**30:.1f}"
    )


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
comin.var_request_add(("var_predict", DOMAIN_ID), lmodexclusive=True)
comin.request_metadata(("var_predict", DOMAIN_ID)).update(zaxis_id=comin.COMIN_ZAXIS_2D)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    ep = [comin.EP_ATM_WRITE_OUTPUT_BEFORE]
    State.icon_var = comin.var_get(ep, (CONFIG.run.icon_var, DOMAIN_ID), comin.COMIN_FLAG_READ | comin.COMIN_FLAG_DEVICE)
    State.predict_var = comin.var_get(ep, ("var_predict", DOMAIN_ID), comin.COMIN_FLAG_WRITE | comin.COMIN_FLAG_DEVICE)
    State.forcing_vars = [comin.var_get(ep, (name, DOMAIN_ID), comin.COMIN_FLAG_READ | comin.COMIN_FLAG_DEVICE) for name in CONFIG.forcings.icon_fields]


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def sample_step():
    """Acts once per sample interval of model time. All ranks see the same
    ICON time, so they take the same branches, as the exchange and DDP need."""
    now = _seconds()
    if State.first_seconds is None:
        State.first_seconds = now
        stride, State.sample_seconds = sample_interval(CONFIG.run.sample_seconds, comin.descrdata_get_timesteplength(DOMAIN_ID))
        comin.print_info(f"sampling every {stride} ICON steps = {State.sample_seconds} s")
        n_dry = int(CONFIG.run.dry_run_seconds // State.sample_seconds)
        comin.print_info(f"dry run: {n_dry} sample steps = {n_dry * stride} ICON steps = {n_dry * State.sample_seconds:.0f} s")
        n_replay = int(CONFIG.forecaster.replay_seconds // State.sample_seconds)
        comin.print_info(
            f"latent replay: keeps the last {n_replay} windows = {n_replay * State.sample_seconds:.0f} s "
            f"({n_replay + CONFIG.forecaster.n_context} latents, each stored once), "
            f"draws {CONFIG.forecaster.replay_batch} per update; windows from the first "
            f"{CONFIG.forecaster.replay_warmup_seconds:.0f} s after the dry run are never replayed"
        )
        kwargs = CONFIG.trainer_kwargs(State.sample_seconds)
        comin.print_info(
            f"key/value memory (last forecaster block): keeps {kwargs['memory_size']} sample steps; a window reads "
            f"the top {kwargs['memory_topk']} entries of each cell's 13-cell window from the "
            f"{CONFIG.forecaster.memory_seconds:.0f} s before its context, through a gate with learning rate "
            f"{kwargs['lr_memory_gate']:g}; the first {kwargs['memory_warmup']} updates are not stored"
            if kwargs["memory_size"]
            else "key/value memory: off"
        )
    if round(now - State.first_seconds) % State.sample_seconds:
        return
    patch = _patch([State.icon_var])

    if State.normalizer is None:
        State.normalizer = RunningMeanStd(patch.shape, int(CONFIG.run.dry_run_seconds // State.sample_seconds), DEVICE)
    if not State.normalizer.update(patch):
        return
    std = State.normalizer.stats.std
    if State.trainer is None:
        comin.print_info(
            f"dry run done at {comin.current_get_datetime()}: per-cell mean/std of {CONFIG.run.icon_var} from "
            f"{State.normalizer.count} samples (rank 0 std {float(std.min()):.3g}-{float(std.max()):.3g}); "
            f"training starts once {CONFIG.forecaster.n_context} sample steps of context are cached"
        )
        State.trainer = _trainer(patch.shape[1], now)

    # Score and write the forecast made one sample step ago, which is valid now.
    fc_rmse = None
    if State.forecast is not None:
        fc_rmse = float((State.forecast - patch)[:N_OWNED].square().mean().sqrt())
        _write(State.forecast)

    torch.cuda.synchronize()
    start = time.perf_counter()
    forcing = State.forcing_norm.normalize(_forcing(now))
    stats = State.trainer.train_step(State.normalizer.normalize(patch), forcing, now)
    then = now + State.sample_seconds
    increment = State.trainer.predict_increment(then, State.forcing_norm.normalize(_forcing(then)))
    torch.cuda.synchronize()
    State.forecast = None if increment is None else patch + increment * std

    if stats:  # the context is full, so a forecast was scored above
        State.log_rows.append({**stats, "fc_rmse": fc_rmse, "train_s": time.perf_counter() - start})
        if len(State.log_rows) >= max(1, int(CONFIG.run.log_seconds // State.sample_seconds)):
            comin.print_info(_log_line())
            State.log_rows.clear()
    State.step += 1
    if State.step % max(1, int(CONFIG.run.save_interval_seconds // State.sample_seconds)) == 0:
        norm = State.normalizer.stats
        extra = {"norm_mean": norm.mean, "norm_std": norm.std}
        extra.update(forcings=FORCING_NAMES, forcing_mean=State.forcing_norm.mean, forcing_std=State.forcing_norm.std)
        save_checkpoint(State.trainer, CHECKPOINT_PATH, rank, State.step, extra)
        comin.print_info(f"checkpoint at step {State.step}, {comin.current_get_datetime()}: {CHECKPOINT_PATH}")


@comin.register_callback(comin.EP_DESTRUCTOR)
def destructor():
    if dist.is_initialized():
        dist.destroy_process_group()
