import comin
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

#----------------------------------------------------------------------------
# GPU check
#----------------------------------------------------------------------------
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
rank = comm.Get_rank() # global MPI rank
world_size = comm.Get_size()

#----------------------------------------------------------------------------
# Pytorch distributed initialization 
# num_io_processes = 1, this will lauch 5 processors for each node, with the last one no GPU
#----------------------------------------------------------------------------
local_rank = int(os.environ.get("SLURM_LOCALID", "-1"))
gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "4"))
has_gpu = local_rank >= 0 and local_rank < gpus_per_node #[0, 1, 2, 3] are GPU-bearing; [4] is IO-only with no GPU]

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



# ---------------------------------------------------------------------------
# Simple MLP: input (batch, 256, 30) -> output (batch, 256, 30)
# 256 = spatial cells per sample, 30 = levels as channels
# Linear is applied independently over the last (channel) dimension.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 512)  # -> (batch, 256, 512)
        self.fc2 = nn.Linear(512, 30)  # -> (batch, 256, 30)

    def forward(self, x):
        # x: (batch, 256, 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


_model = None  # lazily initialised on first callback (DDP-wrapped MLP)
_optimizer = None  # lazily initialised on first callback
_ua_pred_torch = None  # MLP output from previous step; None on first call


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


# ---------------------------------------------------------------------------
# training callbacks
# ---------------------------------------------------------------------------


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    # Non-GPU ranks are excluded from the NCCL group — skip immediately.
    if not has_gpu:
        return

    # build x data from ua
    ua_samples = sample_data(ua)
    comin.print_info(f"x data from {ua_samples.__cuda_array_interface__=}")
    ua_local = to_torch_tensor(ua_samples)
    comin.print_info(f"x: ua_local: shape={ua_local.shape}, dtype={ua_local.dtype}")

    # build y data: use ua (x) as stand-in on the very first call;
    # from the second call onward, use the cached MLP output from the previous step.
    global _ua_pred_torch
    if _ua_pred_torch is None:
        ua_pred_local = ua_local
        comin.print_info(
            f"[rank={rank}] First call: initialising ua_pred_local from ua_local"
        )
    else:
        ua_pred_local = _ua_pred_torch
        comin.print_info(
            f"[rank={rank}] Using cached ua_pred_torch as y, shape={ua_pred_local.shape}"
        )

    # --- training step ---
    global _model, _optimizer
    if _model is None:
        _model, _optimizer = _init_model_and_optimizer()
        comin.print_info(f"[rank={rank}] MLP initialised")

    # cache MLP output with PRE-update params → becomes y for the NEXT step
    with torch.no_grad():
        _ua_pred_torch = _model(ua_local)
    comin.print_info(
        f"[rank={rank}] ua_pred_torch cached (pre-update), shape={_ua_pred_torch.shape}"
    )

    loss = _train_step(_model, _optimizer, ua_local, ua_pred_local)
    comin.print_info(f"[rank={rank}] train loss: {loss:.6f}")
