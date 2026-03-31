import comin
import os
import jax
from jax import dlpack as jdlpack
import numpy as np
import sys
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


from mpi4py import MPI


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
rank = comm.Get_rank()
world_size = comm.Get_size()

num_io_processes = 1
num_calculate_processes = world_size - num_io_processes
io_rank = world_size - 1  # last rank is IO-only, has no GPU

# jax distributed initialization
# Each compute rank owns exactly one GPU: CUDA:0 (its SLURM-assigned device).
# The IO rank (last rank) has no GPU, so local_device_ids=[].
local_device_ids = [0] if rank < io_rank else []
jax.distributed.initialize(local_device_ids=local_device_ids)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    global ta, ua
    ta = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", 1), comin.COMIN_FLAG_READ
    )

    ua = comin.var_get(
        [comin.EP_ATM_NUDGING_BEFORE],
        ("u", 1),
        comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    )


# utility to extract non-halo data and global indices
def no_halo_data(data_array):
    """Extract non-halo data and global indices using xp (CuPy/NumPy)."""
    nc = domain.cells.ncells
    global_idx = xp.asarray(domain.cells.glb_index, dtype=xp.int64) - 1

    # Convert input to xp array and flatten in Fortran order like ICON fields.
    data_xp = xp.asarray(data_array).ravel(order="F")[:nc]
    decomp_xp = xp.asarray(domain.cells.decomp_domain).ravel(order="F")[:nc]

    # Mask for interior cells (where domain == 0)
    halo_mask = decomp_xp == 0

    return data_xp[halo_mask], global_idx[:nc][halo_mask]


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def joo():
    # IO rank has no GPU. It must still call a JAX op to trigger the distributed
    # CPU topology exchange (lazy init) that all 5 processes must participate in.
    # With local_device_ids=[], jax.local_devices() returns [] without CUDA errors.
    if rank >= io_rank:
        _ = jax.local_devices()  # participates in CPU topology rendezvous
        print(f"[rank={rank}] IO-only rank, skipping GPU work.", file=sys.stderr)
        return

    # Mesh over the 4 compute GPUs (JAX processes 0-3, each with 1 GPU).
    mesh = jax.make_mesh((num_calculate_processes,), ("gpu",))
    sharding = NamedSharding(mesh, P("gpu"))

    comin.print_info(f"{ta.__cuda_array_interface__=}")
    ta_cp = xp.asarray(ta)
    comin.print_info(f"[rank={rank}] CuPy device: {ta_cp.device}, shape: {ta_cp.shape}")
    ta_no_halo, _ = no_halo_data(ta_cp)

    local_data = jdlpack.from_dlpack(ta_no_halo)  # JAX array on CUDA:0
    comin.print_info(
        f"[rank={rank}] local_data shape: {local_data.shape}, dtype: {local_data.dtype}"
    )

    # Each compute rank contributes its local shard; JAX assembles the global array.
    array_global = jax.make_array_from_process_local_data(sharding, local_data)
    comin.print_info(
        f"[rank={rank}] array_global shape: {array_global.shape}, "
        f"addressable shards: {len(array_global.addressable_shards)}"
    )


# @comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
# def foo():
#     if rank != 0:  # proc0_shift=1, Use first 1 ranks for training
#         procid = os.getenv("SLURM_PROCID", "?")
#         localid = os.getenv("SLURM_LOCALID", "?")
#         dev = xp.cuda.runtime.getDevice()
#         comin.print_info(f"rank={procid} local_rank={localid} gpu={dev}")
#         total_gpus = jax.local_device_count()
#         comin.print_info(f"Total local GPUs visible to JAX: {total_gpus}")
#         comin.print_info(f"JAX devices: {jax.devices()}")

#     domain_xp = xp.asarray(domain.cells.decomp_domain)
#     nc = domain.cells.ncells
#     mask = domain_xp.ravel(order="F")[:nc] == 0
#     local_prognostic_cells = int(xp.count_nonzero(mask).item())
#     rank_role = "IO" if local_prognostic_cells == 0 else "COMPUTE"
#     print(
#         f"[rank={cpu_rank}] role={rank_role} local_prognostic_cells={local_prognostic_cells}",
#         file=sys.stderr,
#     )
