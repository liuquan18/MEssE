import comin
import os
import jax
from jax import dlpack as jdlpack
import sys

from mpi4py import MPI

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()
world_size = comm.Get_size()

# Explicitly define number of IO ranks for this setup (matches runscript).
NUM_IO_PROCS = 1
COMPUTE_WORLD_SIZE = max(world_size - NUM_IO_PROCS, 0)
IS_IO_RANK = rank >= COMPUTE_WORLD_SIZE


def _parse_visible_cuda_devices():
    """Return CUDA-visible device entries from CUDA_VISIBLE_DEVICES."""
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or cvd.lower() in {"-1", "none", "nodevfiles", "void"}:
        return []
    return [x.strip() for x in cvd.split(",") if x.strip()]


def _get_local_rank():
    for env_name in (
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        value = os.getenv(env_name)
        if value is not None:
            return int(value)
    return 0


JAX_READY = False
JAX_INIT_MSG = "JAX distributed not initialized"
LOCAL_DEVICE_ID = None
CHECK_PRINTED = False


glob = comin.descrdata_get_global()
if glob.has_device:
    comin.print_info(f"{glob.device_name=}")
    comin.print_info(f"{glob.device_vendor=}")
    comin.print_info(f"{glob.device_driver=}")

if IS_IO_RANK:
    import numpy as xp

    DEVICE_SYNC_FLAG = 0
    JAX_INIT_MSG = (
        f"IO rank {rank}: skipping JAX GPU initialization "
        f"(NUM_IO_PROCS={NUM_IO_PROCS})"
    )
elif glob.has_device and "NVIDIA" in glob.device_vendor.upper():
    try:
        comin.print_info("Using cupy!")
        import cupy as xp

        DEVICE_SYNC_FLAG = comin.COMIN_FLAG_DEVICE

        local_rank = _get_local_rank()
        visible_cuda = _parse_visible_cuda_devices()
        if not visible_cuda:
            visible_count = int(xp.cuda.runtime.getDeviceCount())
        else:
            visible_count = len(visible_cuda)

        if visible_count > 0:
            LOCAL_DEVICE_ID = local_rank % visible_count

            # Initialize a compute-only JAX distributed group. The IO rank does not
            # join this group, so we must set num_processes/process_id explicitly.
            coordinator_port = 12000 + (int(os.getenv("SLURM_JOB_ID", "0")) % 20000)
            coordinator_address = os.getenv(
                "JAX_COORDINATOR_ADDRESS", f"127.0.0.1:{coordinator_port}"
            )
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=COMPUTE_WORLD_SIZE,
                process_id=rank,
                local_device_ids=[LOCAL_DEVICE_ID],
                initialization_timeout=300,
            )
            JAX_READY = True
            JAX_INIT_MSG = (
                f"JAX distributed initialized on compute_rank={rank}/{COMPUTE_WORLD_SIZE}, "
                f"local_rank={local_rank}, local_device_id={LOCAL_DEVICE_ID}, "
                f"coord={coordinator_address}, "
                f"CUDA_VISIBLE_DEVICES='{os.getenv('CUDA_VISIBLE_DEVICES', '')}'"
            )
        else:
            JAX_INIT_MSG = (
                f"No CUDA devices visible on compute_rank={rank}/{COMPUTE_WORLD_SIZE}; "
                "JAX GPU path disabled"
            )
    except ImportError as e:
        comin.print_info("Cannot import cupy, falling back to numpy")
        comin.print_info(e)
        comin.print_info(sys.path)
        import numpy as xp

        DEVICE_SYNC_FLAG = 0
        JAX_INIT_MSG = "CuPy import failed; JAX GPU path disabled"
    except Exception as e:
        comin.print_info(f"JAX distributed initialize failed on rank {rank}: {e}")
        import numpy as xp

        DEVICE_SYNC_FLAG = 0
        JAX_INIT_MSG = f"JAX distributed init failed on compute_rank {rank}; disabled"
else:
    comin.print_info("No NVIDIA device found falling back to numpy")
    import numpy as xp

    DEVICE_SYNC_FLAG = 0
    JAX_INIT_MSG = "No NVIDIA device found; JAX GPU path disabled"

print(JAX_INIT_MSG, file=sys.stderr)

domain = comin.descrdata_get_domain(1)


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


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def jax_distributed_check():
    global CHECK_PRINTED
    if CHECK_PRINTED:
        return

    # Check only compute ranks; IO ranks are expected to skip JAX.
    compute_local = 0 if IS_IO_RANK else 1
    compute_total = comm.allreduce(compute_local, op=MPI.SUM)
    compute_ok_local = 1 if (compute_local == 1 and JAX_READY) else 0
    compute_ok = comm.allreduce(compute_ok_local, op=MPI.SUM)

    if rank == 0:
        if compute_ok == compute_total:
            comin.print_info(
                f"JAX_DISTRIBUTED_CHECK: PASS ({compute_ok}/{compute_total} compute ranks initialized, io_ranks={NUM_IO_PROCS})"
            )
        else:
            comin.print_info(
                f"JAX_DISTRIBUTED_CHECK: FAIL ({compute_ok}/{compute_total} compute ranks initialized, io_ranks={NUM_IO_PROCS})"
            )

    CHECK_PRINTED = True


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
def jax_data_2d(arr_cp: xp.ndarray, level: int = 0):
    arr_cp = arr_cp[:, level, :, :, :]
    arr_cp, _ = no_halo_data(arr_cp)

    # to jax array on GPU
    arr_jax = jdlpack.from_dlpack(arr_cp)
    return arr_jax


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def joo():
    if not JAX_READY:
        if rank == 0:
            comin.print_info(
                "Skipping JAX conversion because distributed init is not active"
            )
        return

    # ComIn variable -> CuPy array on GPU
    comin.print_info(f"{ta.__cuda_array_interface__=}")
    ta_cp = xp.asarray(ta)  # xp is cupy in your NVIDIA path
    comin.print_info(
        f"CuPy device: {ta_cp.device}, ptr: {ta_cp.data.ptr}"
    )  # 0:  INFO(cuda_test): ta.__cuda_array_interface__={'shape': (8000, 30, 4, 1, 1), 'typestr': '<f8', 'data': (23455338594304, False), 'version': 3, 'strides': (8, 64000, 1920000, 7680000, 7680000)}
    ta_jax = jdlpack.from_dlpack(ta_cp)
    comin.print_info(
        f"ta_jax device: {ta_jax.device}, shape: {ta_jax.shape}"
    )  # ta_jax device: gpu:0

    ua_cp = xp.asarray(ua)
    comin.print_info(
        f"ua_cp shape: {ua_cp.shape}, device: {ua_cp.device}"
    )  # ua_cp shape: (25600, 30, 4, 1, 1), device: gpu
    ua_jax = jax_data_2d(ua_cp)  # Convert to JAX array on GPU
    comin.print_info(
        f"ua_jax device: {ua_jax.device}, shape: {ua_jax.shape}"
    )  # ua_jax device: gpu:0


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
