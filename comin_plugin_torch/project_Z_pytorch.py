import comin
import dataclasses
import os
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.utils.dlpack as tdlpack
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

from mpi4py import MPI

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
# num_io_processes = 1, this will lauch 5 processors for each node, with the last one no GPU
# ----------------------------------------------------------------------------
local_rank = int(os.environ.get("SLURM_LOCALID", "-1"))
gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "4"))
has_gpu = (
    local_rank >= 0 and local_rank < gpus_per_node
)  # [0, 1, 2, 3] are GPU-bearing; [4] is IO-only with no GPU]

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


# ---------------------------------------------------------------------------
# Simple MLP: input (batch, 256, 31) -> output (batch, 256, 31)
# 30 levels as channels + 1 horizon channel, Linear applied over last dim.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 512)  # -> (batch, 256, 512)
        self.fc2 = nn.Linear(512, 31)  # -> (batch, 256, 31)

    def forward(self, x):
        # x: (batch, 256, 31)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


_model = None  # lazily initialised on first callback (DDP-wrapped MLP)
_optimizer = None  # lazily initialised on first callback
_current_step = 0
_pending_example = None  # ForecastExample | None — the single in-flight forecast


@dataclasses.dataclass
class ForecastExample:
    """A single in-flight forecast linking a past ua snapshot to a future step.

    Attributes
    ----------
    previous_pred : torch.Tensor, shape (batch, 256, nlev+1)
        Model prediction y_t = f_θ(encode(x_t, h)) at creation time, with a
        normalised horizon channel appended (value = horizon / MAX_FORECAST_HORIZON).
    horizon : int
        Number of ICON steps between the input snapshot and the target.
    due_step : int
        Absolute step at which the matching ground-truth ua is available
        (= creation_step + horizon).
    """

    previous_pred: torch.Tensor  # torch tensor on CUDA
    horizon: int
    due_step: int


def _init_model_and_optimizer():
    model = MLP().cuda()
    ddp_model = DDP(model, device_ids=[0])
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    return ddp_model, optimizer


def _train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()


def _sample_horizon() -> int:
    """Sample a random forecast horizon in [1, MAX_FORECAST_HORIZON]."""
    return int(torch.randint(1, MAX_FORECAST_HORIZON + 1, (1,)).item())


def _encode_horizon_channel(arr: torch.Tensor, horizon: int) -> torch.Tensor:
    """Append a normalised horizon channel to a PyTorch tensor.

    Parameters
    ----------
    arr : torch.Tensor, shape (..., nlev)
    horizon : int in [0, MAX_FORECAST_HORIZON]

    Returns
    -------
    torch.Tensor, shape (..., nlev+1)
    """
    horizon_value = horizon / MAX_FORECAST_HORIZON
    horizon_channel = torch.full(
        arr.shape[:-1] + (1,), horizon_value, dtype=torch.float32, device=arr.device
    )
    return torch.cat((arr, horizon_channel), dim=-1)


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


# utility to extract non-halo data and global indices
def no_halo_data(data_array):
    """Extract non-halo data preserving the level dimension.

    Input shape : (nproma, nlev, nblk) or (nproma, nlev, nblk, 1, 1, ...)
    Returns     : data   (n_interior_cells, nlev)
                  global_idx (n_interior_cells,)
    """
    nc = domain.cells.ncells
    global_idx = xp.asarray(domain.cells.glb_index, dtype=xp.int64) - 1

    data_xp = xp.asarray(data_array)
    # squeeze trailing singleton dims: e.g. (nproma, nlev, nblk, 1, 1) → (nproma, nlev, nblk)
    while data_xp.ndim > 3 and data_xp.shape[-1] == 1:
        data_xp = data_xp[..., 0]

    # halo mask derived from decomp_domain (nproma, nblk)
    decomp_xp = xp.asarray(domain.cells.decomp_domain)  # (nproma, nblk)
    halo_mask_1d = decomp_xp.ravel(order="F")[:nc] == 0  # (nc,) True = interior

    if data_xp.ndim == 2:
        # 2-D field (nproma, nblk) – no level dim
        data_cells = data_xp.ravel(order="F")[:nc]
        return data_cells[halo_mask_1d], global_idx[:nc][halo_mask_1d]

    # 3-D field (nproma, nlev, nblk): keep level dimension
    nlev = data_xp.shape[1]
    # transpose to (nproma, nblk, nlev), then F-reshape → (nc, nlev)
    # F-order reshape varies dim-0 (nproma) fastest, matching decomp_domain cell ordering
    data_cells = data_xp.transpose(0, 2, 1).reshape(-1, nlev, order="F")[:nc]
    return data_cells[halo_mask_1d], global_idx[:nc][halo_mask_1d]


# temporal sample strategy: (n_interior, nlev) -> (batch, sample_size, nlev)
def sample_data(arr, sample_size=256):
    arr_cp = xp.asarray(arr)
    comin.print_info(
        f"[rank={rank}] CuPy device: {arr_cp.device}, shape: {arr_cp.shape}"
    )
    # no_halo_data returns (n_interior_cells, nlev)
    arr_no_halo, _ = no_halo_data(arr_cp)
    comin.print_info(f"[rank={rank}] no_halo_data output shape: {arr_no_halo.shape}")
    # arr_no_halo: (n_interior, nlev) -> (batch, sample_size, nlev)
    nlev = arr_no_halo.shape[1]
    n_batch = arr_no_halo.shape[0] // sample_size
    arr_samples = arr_no_halo[: n_batch * sample_size].reshape(
        n_batch, sample_size, nlev
    )
    return arr_samples


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
    ua_current=None,
    current_step=0,
):
    """Utility to run the training callback logic for a single step, with options to enable/disable stages."""

    if mode == "waiting":
        comin.print_info(
            f"[rank={rank}] step={current_step}, waiting "
            f"(due at step={_pending_example.due_step}, horizon={_pending_example.horizon})"
        )
        return _pending_example, _model, _optimizer

    if mode == "initial" and _pending_example is None:
        _model, _optimizer = _init_model_and_optimizer()
        comin.print_info(f"[rank={rank}] MLP initialised")

    if mode == "training" and _pending_example is not None:
        x_local = (
            _pending_example.previous_pred
        )  # prediction from previous step (y_t) (31ch)
        y_local = _encode_horizon_channel(
            ua_current, _pending_example.horizon
        )  # current ua as ground truth (x_{t+h}) (31ch)
        loss = _train_step(_model, _optimizer, x_local, y_local)
        comin.print_info(f"trained horizon={_pending_example.horizon}, loss={loss:.6f}")

    if predicting:
        horizon = _sample_horizon()
        x_enc = _encode_horizon_channel(ua_current, horizon)  # 31ch input
        with torch.no_grad():
            ua_predict = _model(x_enc)  # 31ch prediction

        _pending_example = ForecastExample(
            previous_pred=ua_predict,
            horizon=horizon,
            due_step=current_step + horizon,
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
    ua_samples = sample_data(ua)
    comin.print_info(f"x data from {ua_samples.__cuda_array_interface__=}")
    ua_local = to_torch_tensor(ua_samples)
    comin.print_info(f"x: ua_local: shape={ua_local.shape}, dtype={ua_local.dtype}")

    # ---- step0: initializing and predicting --
    if current_step == 0 and _pending_example is None:
        _pending_example, _model, _optimizer = process_step(
            mode="initial",
            predicting=True,
            ua_current=ua_local,
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
            ua_current=ua_local,
            current_step=current_step,
        )
