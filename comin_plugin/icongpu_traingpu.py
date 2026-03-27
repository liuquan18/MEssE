import comin
import os
import jax
import jax.numpy as jnp
from jax import dlpack as jdlpack
import sys

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
# %%
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()
world_size = comm.Get_size()


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

# function data build
def jax_data_2d(arr_cp: xp.ndarray, level: int=0):
    arr_cp = arr_cp[:, level, :, :, :]
    arr_cp, _ = no_halo_data(arr_cp)

    # to jax array on GPU
    arr_jax = jdlpack.from_dlpack(arr_cp)
    return arr_jax


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def joo():
    # ComIn variable -> CuPy array on GPU
    comin.print_info(f"{ta.__cuda_array_interface__=}")
    ta_cp = xp.asarray(ta)  # xp is cupy in your NVIDIA path
    comin.print_info(f"CuPy device: {ta_cp.device}, ptr: {ta_cp.data.ptr}") #0:  INFO(cuda_test): ta.__cuda_array_interface__={'shape': (8000, 30, 4, 1, 1), 'typestr': '<f8', 'data': (23455338594304, False), 'version': 3, 'strides': (8, 64000, 1920000, 7680000, 7680000)}
    ta_jax = jdlpack.from_dlpack(ta_cp)
    comin.print_info(f"ta_jax device: {ta_jax.device_buffer.device()}, shape: {ta_jax.shape}")  # ta_jax device: gpu:0


    ua_cp = xp.asarray(ua)
    comin.print_info(f"ua_cp shape: {ua_cp.shape}, device: {ua_cp.device}") # ua_cp shape: (25600, 30, 4, 1, 1), device: gpu
    ua_jax = jax_data_2d(ua_cp)  # Convert to JAX array on GPU
    comin.print_info(f"ua_jax device: {ua_jax.device_buffer.device()}, shape: {ua_jax.shape}")  # ua_jax device: gpu:0


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
