import comin
import os
import jax
import jax.numpy as jnp
from jax import dlpack as jdlpack
import numpy as np
import sys
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax.linen as nn
import optax
from flax.training import train_state as flax_train_state


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


# ---------------------------------------------------------------------------
# Simple MLP: input (batch, 256) -> output (batch, 256)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        return x


_train_state = None  # lazily initialised on first callback


def _init_train_state(mesh):
    model = MLP()
    replicated = NamedSharding(mesh, P())  # params replicated on all GPUs
    rng = jax.random.PRNGKey(0)
    params = jax.jit(model.init, out_shardings=replicated)(rng, jnp.ones((1, 256)))
    tx = optax.adam(1e-3)
    return flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


_train_state = None  # lazily initialised on first callback
_ua_pred_jax = None  # MLP output from previous step; None on first call


@jax.jit
def _train_step(state, x, y):
    def loss_fn(params):
        pred = state.apply_fn(params, x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ---------------------------------------------------------------------------
# constructor callback: setup variables and data extraction utilities
# ---------------------------------------------------------------------------

# primary constructor callback to register new variables
var_descriptor = ("ua_pred", 1)
comin.var_request_add(
    var_descriptor, lmodexclusive=True
)  # request variable from icon, with write
comin.metadata_set(
    var_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_3D,
    long_name="Predicted zonal wind from MLP",
    units="m/s",
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

    # ua_pred = comin.var_get(
    #     [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
    #     ("ua_pred", 1),
    #     comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    # )


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


def sample_data(arr, level=0, sample_size=256):
    arr_cp = xp.asarray(arr)
    comin.print_info(
        f"[rank={rank}] CuPy device: {arr_cp.device}, shape: {arr_cp.shape}"
    )
    arr_level = arr_cp[:, level, ...]
    arr_no_halo, _ = no_halo_data(arr_level)

    arr_samples = arr_no_halo.reshape(-1, sample_size)
    return arr_samples


def global_jax_array(arr: xp.ndarray, sharding) -> jax.Array:
    local_data = jdlpack.from_dlpack(arr)  # JAX array on CUDA:0
    comin.print_info(
        f"[rank={rank}] local_data shape: {local_data.shape}, dtype: {local_data.dtype}"
    )

    # Each compute rank contributes its local shard; JAX assembles the global array.
    array_global = jax.make_array_from_process_local_data(sharding, local_data)

    return array_global


# ---------------------------------------------------------------------------
# training callbacks
# ---------------------------------------------------------------------------


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    # IO rank has no GPU. It must still call a JAX op to trigger the distributed
    # CPU topology exchange (lazy init) that all 5 processes must participate in.
    # With local_device_ids=[], jax.local_devices() returns [] without CUDA errors.
    if rank >= io_rank:
        _ = jax.local_devices()  # participates in CPU topology rendezvous
        # print(f"[rank={rank}] IO-only rank, skipping GPU work.", file=sys.stderr)
        return

    # Mesh over the 4 compute GPUs (JAX processes 0-3, each with 1 GPU).
    mesh = jax.make_mesh((num_calculate_processes,), ("gpu",))
    sharding = NamedSharding(mesh, P("gpu"))

    # build x data from ua
    ua_samples = sample_data(ua)
    comin.print_info(f"x data from {ua_samples.__cuda_array_interface__=}")
    ua_global = global_jax_array(ua_samples, sharding)
    comin.print_info(
        f"x: ua_global: shape={ua_global.shape}, dtype={ua_global.dtype}, sharding={ua_global.sharding}"
    )

    # build y data: use ua (x) as stand-in on the very first call;
    # from the second call onward, use the cached MLP output from the previous step.
    global _ua_pred_jax
    if _ua_pred_jax is None:
        ua_pred_global = ua_global
        comin.print_info(
            f"[rank={rank}] First call: initialising ua_pred_global from ua_global"
        )
    else:
        ua_pred_global = _ua_pred_jax
        comin.print_info(
            f"[rank={rank}] Using cached ua_pred_jax as y, shape={ua_pred_global.shape}"
        )

    # --- training step ---
    global _train_state
    if _train_state is None:
        _train_state = _init_train_state(mesh)
        comin.print_info(f"[rank={rank}] MLP initialised")

    # cache MLP output with PRE-update params → becomes y for the NEXT step
    # (must be done before _train_step so y and pred use different params)
    _ua_pred_jax = _train_state.apply_fn(_train_state.params, ua_global)
    comin.print_info(
        f"[rank={rank}] ua_pred_jax cached (pre-update), shape={_ua_pred_jax.shape}"
    )

    _train_state, loss = _train_step(_train_state, ua_global, ua_pred_global)
    comin.print_info(f"[rank={rank}] train loss: {loss:.6f}")
