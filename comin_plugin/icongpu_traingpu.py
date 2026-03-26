import comin
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import dlpack as jdlpack

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



@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def sec_ctor():
    global ta
    ta = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", 1), comin.COMIN_FLAG_READ
    )




@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def joo():
    # ComIn variable -> CuPy array on GPU

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
    procid = os.getenv("SLURM_PROCID", "?")
    localid = os.getenv("SLURM_LOCALID", "?")
    dev = xp.cuda.runtime.getDevice()
    comin.print_info(f"rank={procid} local_rank={localid} gpu={dev}")
    total_gpus = jax.local_device_count()
    comin.print_info(f"Total local GPUs visible to JAX: {total_gpus}")