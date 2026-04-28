import comin
import dataclasses
import datetime
import math
import os
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.dlpack as tdlpack
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import cftime

from field_space_attention_wrapper import FieldSpaceAttentionWrapper

from mpi4py import MPI
import earth2grid.healpix as e2g_healpix
from earth2grid._regrid import KNNS2Interpolator

# ----------------------------------------------------------------------------
# GPU check
# ----------------------------------------------------------------------------
glob = comin.descrdata_get_global()
if glob.has_device:
    comin.print_info(f"{glob.device_name=}")
    comin.print_info(f"{glob.device_vendor=}")
    comin.print_info(f"{glob.device_driver=}")

if glob.has_device and "NVIDIA" in glob.device_vendor.upper():
    try:
        comin.print_info("Using cupy!")
        import cupy as xp

        DEVICE_SYNC_FLAG = comin.COMIN_FLAG_DEVICE
    except ImportError as e:
        comin.print_info("Cannot import cupy, falling back to numpy")
        comin.print_info(e)
        import sys

        comin.print_info(sys.path)
        import numpy as xp

        DEVICE_SYNC_FLAG = 0
else:
    comin.print_info("No NVIDIA device found falling back to numpy")
    import numpy as xp

    DEVICE_SYNC_FLAG = 0


domain = comin.descrdata_get_domain(1)

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()  # global MPI rank
world_size = comm.Get_size()

# ----------------------------------------------------------------------------
# Pytorch distributed initialization
# Detect GPU-bearing ranks via CUDA_VISIBLE_DEVICES: the run_wrapper (levante.sh)
# sets CUDA_VISIBLE_DEVICES to a single GPU index (e.g. "0", "1") for compute ranks
# and leaves it unset (or as all GPUs) for IO ranks.  Using SLURM_LOCALID < GPUS_ON_NODE
# is NOT reliable because the IO rank also has a valid local rank within the GPU count.
# ----------------------------------------------------------------------------
_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
has_gpu = _cuda_vis.isdigit()  # True only when wrapper set exactly one GPU index

# Count how many MPI ranks are GPU-bearing globally.
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)
num_no_gpu_processes = world_size - num_calculate_processes

compute_rank = None  # None for the IO rank

if has_gpu:
    # Split off a communicator that contains only GPU-bearing ranks.
    compute_comm = comm.Split(color=0, key=rank)
    compute_rank = compute_comm.Get_rank()

    # All compute ranks agree on MASTER_ADDR (hostname of compute rank 0).
    if compute_rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None
    master_addr = compute_comm.bcast(master_addr, root=0)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    # For GPU-bearing ranks, run_wrapper sets a single CUDA_VISIBLE_DEVICES entry.
    # Therefore the process-local CUDA index is always 0.
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
    # Non-GPU ranks: split into a non-NCCL communicator.
    _ = comm.Split(color=1, key=rank)


MAX_FORECAST_HORIZON = 10


# Model is lazily initialised on first callback.
_model = None
_optimizer = None  # lazily initialised on first callback
_current_step = 0
_pending_example = None  # ForecastExample | None — the single in-flight forecast

# ---------------------------------------------------------------------------
# HEALPix regridding globals (initialised once in sec_ctor for GPU ranks)
# ---------------------------------------------------------------------------
HPX_LEVEL = 6
_hpx_regridder = None  # Regridder: (nc, nlev) -> (n_owned, nlev)
_n_owned_pixels = 0  # total owned pixels = _faces_per_rank * nside²
_faces_per_rank = 0  # number of HEALPix faces owned by this compute rank
_first_face = 0  # first global HEALPix face id owned by this rank

# ---------------------------------------------------------------------------
# Calendar embedding globals
# ---------------------------------------------------------------------------
CALENDAR_EMBED_CHANNELS = 4  # FrequencyEmbedding channels per signal → 4*C total
_calendar_embedding = (
    None  # CalendarEmbedding | None — initialised in _init_hpx_regridding
)


@dataclasses.dataclass
class ForecastExample:
    """A single in-flight forecast linking a past ua snapshot to a future step.

    Attributes
    ----------
    previous_pred : torch.Tensor, shape (faces_per_rank, nlev, nside, nside)
        Predicted ua tensor at creation time; used as x at training time.
    horizon : int
        Number of ICON steps between the input snapshot and the target.
    due_step : int
        Absolute step at which the matching ground-truth ua is available
        (= creation_step + horizon).
    """

    previous_pred: torch.Tensor  # shape (faces_per_rank, nlev, nside, nside) on CUDA
    horizon: int
    due_step: int
    day_of_year: float  # days since 1 Jan at prediction time t
    second_of_day: float  # seconds since midnight UTC at prediction time t


@dataclasses.dataclass
class ModelInput:
    """A single model input snapshot capturing wind data and calendar context.

    Attributes
    ----------
    ua_hp : torch.Tensor, shape (faces_per_rank, nlev, nside, nside)
        HEALPix-regridded zonal wind on CUDA, output of to_hpx_faces().
    day_of_year : float
        Days elapsed since 1 January of the current year (0 – ~365.25).
    second_of_day : float
        Seconds elapsed since midnight UTC (0 – 86400).
    """

    ua_hp: torch.Tensor  # shape (faces_per_rank, nlev, nside, nside) on CUDA
    day_of_year: float
    second_of_day: float


# ---------------------------------------------------------------------------
# Calendar helpers (cBottle-style)
# ---------------------------------------------------------------------------


def _second_of_day(time) -> float:
    """Returns the seconds elapsed since the start of the day."""
    begin_of_day = time.replace(hour=0, minute=0, second=0)
    return (time - begin_of_day).total_seconds()


def _get_calendar_inputs(time: cftime.DatetimeGregorian):
    """Calculate second_of_day and day_of_year from a cftime datetime.

    Mirrors the logic in cbottle/datasets/dataset_3d.py.

    Returns
    -------
    day_of_year : float  — days since 1 Jan of current year (0 – ~365.25)
    second_of_day : float — seconds since midnight UTC (0 – 86400)
    """
    day_start = time.replace(hour=0, minute=0, second=0)
    year_start = day_start.replace(month=1, day=1)
    sec_of_day = (time - day_start).total_seconds()
    day_of_yr = (time - year_start).total_seconds() / 86400.0
    return day_of_yr, sec_of_day


def _parse_icon_datetime(iso_str: str) -> cftime.DatetimeGregorian:
    """Parse the ISO 8601 string returned by comin.current_get_datetime().

    ICON returns strings like ``2020-01-20T00:00:00.000``.
    """
    # Strip sub-second precision and trailing Z if present, then parse.
    clean = iso_str.split(".")[0].rstrip("Z")
    dt = datetime.datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
    return cftime.DatetimeGregorian(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )


# ---------------------------------------------------------------------------
# Calendar embedding modules (cBottle-style)
# ---------------------------------------------------------------------------


class FrequencyEmbedding(nn.Module):
    """Periodic embedding for inputs normalised to [0, 1].

    Applies sin/cos at integer frequencies 1 … num_channels, producing
    2 * num_channels output channels.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, num_channels + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (faces, 1, H, W)  →  out: (faces, 2*num_channels, H, W)
        freqs = self.freqs.view(1, -1, 1, 1)  # (1, C, 1, 1)
        x_scaled = x * (2 * math.pi * freqs)  # (faces, C, H, W)  — broadcasts on dim 1
        return torch.cat([x_scaled.cos(), x_scaled.sin()], dim=1)  # (faces, 2C, H, W)


class CalendarEmbedding(nn.Module):
    """Embed time-of-day and day-of-year using longitude-corrected local solar time.

    Mirrors CalendarEmbedding in cbottle.

    Parameters
    ----------
    lon : torch.Tensor, shape (faces_per_rank, nside, nside)
        Longitude in degrees for each owned HEALPix pixel.
    embed_channels : int
        Number of frequency channels per temporal signal.
        Output has 4 * embed_channels channels total.
    """

    def __init__(self, lon: torch.Tensor, embed_channels: int):
        super().__init__()
        self.register_buffer("lon", lon)  # (faces, nside, nside)
        self.embed_second = FrequencyEmbedding(embed_channels)
        self.embed_day = FrequencyEmbedding(embed_channels)

    def forward(self, day_of_year: float, second_of_day: float) -> torch.Tensor:
        # self.lon: (faces, nside, nside)
        # Convert UTC second_of_day → local solar time using longitude.
        local_time = (
            second_of_day + self.lon * 86400.0 / 360.0
        ) % 86400.0  # (faces, nside, nside)
        local_time = local_time.unsqueeze(1)  # (faces, 1, nside, nside)
        doy = torch.full_like(local_time, day_of_year)  # (faces, 1, nside, nside)
        a = self.embed_second(local_time / 86400.0)  # (faces, 2C, nside, nside)
        b = self.embed_day((doy / 365.25) % 1.0)  # (faces, 2C, nside, nside)
        return torch.cat([a, b], dim=1)  # (faces, 4C, nside, nside)


def _init_model_and_optimizer(nlev: int, calendar_channels: int):
    model = FieldSpaceAttentionWrapper(
        nlev=nlev,
        calendar_channels=calendar_channels,
        hpx_level=HPX_LEVEL,
        first_face=_first_face,
        faces_per_rank=_faces_per_rank,
    ).cuda()
    ddp_model = DDP(model, device_ids=[0])
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    return ddp_model, optimizer


def _train_step(
    model,
    optimizer,
    x_ua: torch.Tensor,
    x_calendar: torch.Tensor,
    y_ua: torch.Tensor,
):
    if x_ua.shape != y_ua.shape:
        raise ValueError(f"x_ua shape {x_ua.shape} must match y_ua shape {y_ua.shape}")
    if x_ua.shape[0] != x_calendar.shape[0] or x_ua.shape[-2:] != x_calendar.shape[-2:]:
        raise ValueError(
            "x_calendar must match x_ua in faces/spatial dims, got "
            f"x_ua={x_ua.shape}, x_calendar={x_calendar.shape}"
        )

    optimizer.zero_grad()
    pred = model(x_ua, x_calendar)
    loss = torch.mean((pred - y_ua) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()


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


# ---------------------------------------------------------------------------
# data constructor: setup variables and data extraction utilities
# ---------------------------------------------------------------------------

# primary constructor callback to register new variables (only if ua_pred is to be written)
# var_descriptor = ("ua_pred", 1)
# comin.var_request_add(
#     var_descriptor, lmodexclusive=True
# )  # request variable from icon, with write
# comin.metadata_set(
#     var_descriptor,
#     zaxis_id=comin.COMIN_ZAXIS_3D,
#     long_name="Predicted zonal wind from MLP",
#     units="m/s",
# )


def _init_hpx_regridding():
    """Build and cache the per-rank ICON→HEALPix nearest-neighbour regrid operator.

    Called once from sec_ctor on GPU-bearing ranks.

    Ownership is face-based: HEALPix has exactly 12 base faces, and each
    compute rank owns (12 // n_compute) consecutive faces in NEST order.
    In NEST ordering the pixel indices for face f are the contiguous range
    [f*nside², (f+1)*nside²), so no global communication is needed.

    ICON's halo exchange ensures that each rank already holds the field values
    for cells belonging to neighbouring ranks near face boundaries, so nearest-
    neighbour sampling into owned pixels is fully local.

    Requires: 12 % n_compute == 0  (valid GPU counts: 1, 2, 3, 4, 6, 12)
    """
    global _hpx_regridder, _n_owned_pixels, _faces_per_rank, _first_face

    nc = domain.cells.ncells
    n_compute = compute_comm.Get_size()

    if 12 % n_compute != 0:
        comin.print_info(
            f"[rank={rank}] WARNING: n_compute={n_compute} does not divide 12; "
            "face-based HEALPix ownership is uneven — regrid disabled"
        )
        return

    # --- face assignment: each rank owns (12 // n_compute) consecutive faces ---
    nside = 2**HPX_LEVEL  # 64 for level 6
    faces_per_rank = 12 // n_compute
    first_face = compute_rank * faces_per_rank
    last_face = first_face + faces_per_rank  # exclusive

    # In NEST ordering face f occupies pixels [f*nside², (f+1)*nside²)
    owned_pixel_ids = np.arange(
        first_face * nside**2, last_face * nside**2, dtype=np.int64
    )
    _n_owned_pixels = len(owned_pixel_ids)
    _faces_per_rank = faces_per_rank
    _first_face = first_face

    comin.print_info(
        f"[rank={rank}/cr={compute_rank}] HPX init: faces {first_face}–{last_face-1}, "
        f"{_n_owned_pixels} pixels"
    )

    # --- owned pixel centres (degrees) via NEST-ordered grid ---
    grid = e2g_healpix.Grid(level=HPX_LEVEL, pixel_order=e2g_healpix.PixelOrder.NEST)
    owned_ids_t = torch.tensor(owned_pixel_ids, dtype=torch.long)
    owned_lon, owned_lat = grid.pix2ang(owned_ids_t)  # both in degrees, float64

    # --- cell coordinates in degrees (all local cells: interior + halo) ---
    # domain.cells.clon/clat may be 2D (nproma, nblk) in Fortran block layout;
    # ravel with Fortran order so cells are indexed column-major (nproma fastest),
    # then take the first nc elements to get a 1D array matching ncells.
    cell_lon = np.rad2deg(
        np.asarray(domain.cells.clon, dtype=np.float64).ravel(order="F")
    )[:nc]
    cell_lat = np.rad2deg(
        np.asarray(domain.cells.clat, dtype=np.float64).ravel(order="F")
    )[:nc]

    # --- nearest-neighbour regridder (k=1): no weighting needed ---
    src_lon_t = torch.tensor(cell_lon, dtype=torch.float32)
    src_lat_t = torch.tensor(cell_lat, dtype=torch.float32)

    regridder = KNNS2Interpolator(
        src_lon=src_lon_t.flatten(),
        src_lat=src_lat_t.flatten(),
        dest_lon=owned_lon.float().flatten(),
        dest_lat=owned_lat.float().flatten(),
        k=1,
    )
    _hpx_regridder = regridder.cuda()
    comin.print_info(
        f"[rank={rank}] HPX regridder ready: {nc} src cells → {_n_owned_pixels} owned pixels"
    )

    # --- calendar embedding (no learnable params; uses owned pixel longitudes) ---
    global _calendar_embedding
    lon_faces = owned_lon.float().cuda().view(faces_per_rank, nside, nside)
    _calendar_embedding = CalendarEmbedding(
        lon=lon_faces, embed_channels=CALENDAR_EMBED_CHANNELS
    ).cuda()
    comin.print_info(
        f"[rank={rank}] CalendarEmbedding ready: "
        f"{4 * CALENDAR_EMBED_CHANNELS} channels, lon shape={lon_faces.shape}"
    )


# second constructor callback to get access to variables created by icon
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    global ua  # , ua_pred

    ua = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        ("u", 1),
        comin.COMIN_FLAG_READ | DEVICE_SYNC_FLAG,
    )

    # this is to save the prediction to icon, but writing a PyTorch tensor back to the
    # ICON buffer (with halo re-insertion) is non-trivial,
    # ua_pred = comin.var_get(
    #     [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
    #     ("ua_pred", 1),
    #     comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    # )

    if has_gpu:
        _init_hpx_regridding()


def all_local_data(data_array):
    """Extract all local cells (interior + halo) preserving the level dimension.

    Input shape : (nproma, nlev, nblk) or (nproma, nlev, nblk, 1, 1, ...)
    Returns     : data (n_local_cells, nlev)   — includes halo cells as donors
    """
    nc = domain.cells.ncells
    data_xp = xp.asarray(data_array)
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]
    if data_xp.ndim == 2:
        return data_xp.ravel(order="F")[:nc]
    nlev = data_xp.shape[1]
    return data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]  # (nc, nlev)


def regrid_to_healpix(data_array) -> torch.Tensor:
    """Regrid ICON cells to owned HEALPix pixels.

    Returns: (n_owned, nlev) float32 tensor on CUDA, or None if not initialised.
    """
    if _hpx_regridder is None or _n_owned_pixels == 0:
        return None
    cells = all_local_data(data_array)  # (nc, nlev), cupy on GPU
    cells_t = tdlpack.from_dlpack(cells).float()  # (nc, nlev) torch CUDA tensor
    # Regridder.forward expects (*, n_src) → produces (*, n_dest)
    owned = _hpx_regridder(cells_t.T)  # (nlev, nc) → (nlev, n_owned)
    return owned.T.contiguous()  # (n_owned, nlev)


def to_hpx_faces(owned_vals: torch.Tensor) -> torch.Tensor:
    """Reshape (n_owned, nlev) → (faces_per_rank, nlev, nside, nside).

    In NEST ordering each face's nside² pixels are stored contiguously and
    follow a Z-curve that maps exactly to a 2D (nside, nside) spatial grid,
    giving a proper image tensor that Conv2d can exploit.
    """
    nside = 2**HPX_LEVEL  # 64 for level 6
    n_owned, nlev = owned_vals.shape
    faces = n_owned // (nside * nside)  # = faces_per_rank
    # (n_owned, nlev) → (faces, nside, nside, nlev) → (faces, nlev, nside, nside)
    return (
        owned_vals.reshape(faces, nside, nside, nlev).permute(0, 3, 1, 2).contiguous()
    )


def to_torch_tensor(arr) -> torch.Tensor:
    """Convert a CuPy (or DLPack-compatible) array to a PyTorch CUDA tensor."""
    tensor = tdlpack.from_dlpack(arr)  # zero-copy via DLPack
    comin.print_info(
        f"[rank={rank}] to_torch_tensor: shape={tensor.shape}, dtype={tensor.dtype}"
    )
    return tensor.float()  # ensure float32 for the MLP


def process_step(
    mode,
    predicting=False,
    _pending_example=None,
    _model=None,
    _optimizer=None,
    model_input=None,
    current_step=0,
):
    """Utility to run the training callback logic for a single step, with options to enable/disable stages.

    Parameters
    ----------
    model_input : ModelInput | None
        Current-step ua snapshot with calendar context.  Required in all modes
        except "waiting".
    """

    if mode == "waiting":
        comin.print_info(
            f"[rank={rank}] step={current_step}, waiting "
            f"(due at step={_pending_example.due_step}, horizon={_pending_example.horizon})"
        )
        return _pending_example, _model, _optimizer

    if mode == "initial" and _pending_example is None:
        # Model predicts ua while receiving calendar as a separate input branch.
        nlev = model_input.ua_hp.shape[1]
        cal_channels = 4 * CALENDAR_EMBED_CHANNELS
        _model, _optimizer = _init_model_and_optimizer(
            nlev=nlev,
            calendar_channels=cal_channels,
        )
        comin.print_info(
            f"[rank={rank}] Field-space model initialised, nlev={nlev}, "
            f"calendar_channels={cal_channels}, first_face={_first_face}"
        )

    if mode == "training" and _pending_example is not None:
        x_ua = _pending_example.previous_pred
        x_cal = _calendar_embedding(
            _pending_example.day_of_year,
            _pending_example.second_of_day,
        )
        y_ua = model_input.ua_hp
        loss = _train_step(_model, _optimizer, x_ua, x_cal, y_ua)
        comin.print_info(f"trained horizon={_pending_example.horizon}, loss={loss:.6f}")

    if predicting:
        # horizon = _sample_horizon()
        horizon = 30  # keep fixed horizon in the first field-space integration pass
        cal_pred = _calendar_embedding(
            model_input.day_of_year, model_input.second_of_day
        )
        with torch.no_grad():
            ua_predict = _model(model_input.ua_hp, cal_pred)

        if ua_predict.shape != model_input.ua_hp.shape:
            raise RuntimeError(
                f"prediction shape {ua_predict.shape} does not match ua shape {model_input.ua_hp.shape}"
            )

        _pending_example = ForecastExample(
            previous_pred=ua_predict,
            horizon=horizon,
            due_step=current_step + horizon,
            day_of_year=model_input.day_of_year,
            second_of_day=model_input.second_of_day,
        )
        comin.print_info(
            f"enqueued horizon={horizon}, due_step={_pending_example.due_step}"
        )

    return _pending_example, _model, _optimizer


# ---------------------------------------------------------------------------
# training callbacks
# ---------------------------------------------------------------------------


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online training callback, called by ICON at every time step.

    Three modes, depending on the time step:
    WAITING         – Between t and t+h, a prediction is in flight but not yet due;
                    return immediately with zero GPU overhead.
    INITIALIZING    – At step 0, no prediction is in flight; initialise the model
                    later a prediction is made using the initialized parameters.
    TRAINING        – When the current step matches the due step (t+h),
                    train on the pair (previous_pred → encode(ua_current, h))

    The PREDICTION stage runs at both step 0 (after INITIALIZING) and at due steps (after TRAINING).
    PREDICTION      – After training (or initialisation at step 0), predict a new
                    ua snapshot to be used as the "previous_pred" for a future step.
    """
    if not has_gpu:
        return

    global _current_step, _pending_example, _model, _optimizer

    current_step = _current_step
    _current_step += 1

    # ---- between step t and t+h, waiting ---------------------
    if _pending_example is not None and _pending_example.due_step != current_step:
        _, _, _ = process_step(
            mode="waiting",
            predicting=False,
            _pending_example=_pending_example,
            current_step=current_step,
        )
        return

    # ---- data preparation --------
    # only prepare data when needed
    ua_hpx = regrid_to_healpix(ua)
    if ua_hpx is None:
        comin.print_info(
            f"[rank={rank}] HPX regrid not ready, skipping step {current_step}"
        )
        return
    comin.print_info(
        f"[rank={rank}] step={current_step} HPX output: shape={ua_hpx.shape}, dtype={ua_hpx.dtype}"
    )
    ua_local = to_hpx_faces(ua_hpx)  # (faces_per_rank, nlev, nside, nside)
    comin.print_info(f"[rank={rank}] HPX faces: shape={ua_local.shape}")

    # ---- build ModelInput (ua + calendar context) ----
    _icon_time = _parse_icon_datetime(comin.current_get_datetime())
    _day_of_yr, _sec_of_day = _get_calendar_inputs(_icon_time)
    model_input = ModelInput(
        ua_hp=ua_local,
        day_of_year=_day_of_yr,
        second_of_day=_sec_of_day,
    )
    comin.print_info(
        f"[rank={rank}] ModelInput: day_of_year={model_input.day_of_year:.3f}, "
        f"second_of_day={model_input.second_of_day:.1f}"
    )

    # ---- step0: initializing and predicting --
    if current_step == 0 and _pending_example is None:
        _pending_example, _model, _optimizer = process_step(
            mode="initial",
            predicting=True,
            model_input=model_input,
            current_step=current_step,
        )

    # ---- due step (t+h): training (previous model prediction -> encode(ua_current, h)) and predicting --
    else:
        _pending_example, _model, _optimizer = process_step(
            mode="training",
            predicting=True,
            _pending_example=_pending_example,
            _model=_model,
            _optimizer=_optimizer,
            model_input=model_input,
            current_step=current_step,
        )
