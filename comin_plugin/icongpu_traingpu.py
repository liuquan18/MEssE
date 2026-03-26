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
cpu_rank = comm.Get_rank()


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    global ta
    ta = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", 1), comin.COMIN_FLAG_READ
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def joo():
    # ComIn variable -> CuPy array on GPU
    if cpu_rank != 0:    # proc0_shift=1, Use first 1 ranks for training
        comin.print_info(f"{ta.__cuda_array_interface__=}")
        ta_cp = xp.asarray(ta)  # xp is cupy in your NVIDIA path
        comin.print_info(f"CuPy device: {ta_cp.device}, ptr: {ta_cp.data.ptr}")

        # CuPy -> JAX (GPU) via DLPack
        ta_jax = jdlpack.from_dlpack(ta_cp)
        comin.print_info(f"JAX device: {ta_jax.device}")

        # Simple JAX compute on GPU
        tas_jax = ta_jax[:, -1, :, 0, 0]
        mean_tas = jnp.mean(tas_jax)

        # Force execution so you really test runtime/device path
        mean_tas.block_until_ready()
        comin.print_info(f"JAX mean surface temp = {float(mean_tas)}")


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def foo():
    if cpu_rank != 0:    # proc0_shift=1, Use first 1 ranks for training
        procid = os.getenv("SLURM_PROCID", "?")
        localid = os.getenv("SLURM_LOCALID", "?")
        dev = xp.cuda.runtime.getDevice()
        comin.print_info(f"rank={procid} local_rank={localid} gpu={dev}")
        total_gpus = jax.local_device_count()
        comin.print_info(f"Total local GPUs visible to JAX: {total_gpus}")
        comin.print_info(f"JAX devices: {jax.devices()}")

    domain_xp = xp.asarray(domain.cells.decomp_domain)
    nc = domain.cells.ncells
    mask = domain_xp.ravel(order="F")[:nc] == 0
    local_prognostic_cells = int(xp.count_nonzero(mask).item())
    rank_role = "IO" if local_prognostic_cells == 0 else "COMPUTE"
    print(
        f"[rank={cpu_rank}] role={rank_role} local_prognostic_cells={local_prognostic_cells}",
        file=sys.stderr,
    )
