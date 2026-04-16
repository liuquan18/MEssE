import comin
import dataclasses
import os
import socket

# Prevent JAX from pre-allocating ~90% of GPU memory, which would leave
# almost nothing for ICON's OpenACC allocations.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import dlpack as jdlpack
import sys
from jax.sharding import NamedSharding, PartitionSpec as P
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

# ICON can launch more MPI tasks per node than physical GPUs (e.g. 5 tasks vs 4 GPUs).
# We use the SLURM local rank to identify which tasks are GPU-bearing.
local_rank = int(os.environ.get("SLURM_LOCALID", "-1"))
gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "4"))
has_gpu = local_rank >= 0 and local_rank < gpus_per_node

# Count how many MPI ranks are GPU-bearing globally.
num_calculate_processes = comm.allreduce(1 if has_gpu else 0, op=MPI.SUM)

print(
    f"[rank={rank}] local_rank={local_rank}, gpus_per_node={gpus_per_node}, "
    f"has_gpu={has_gpu}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    file=sys.stderr,
)

# ---------------------------------------------------------------------------
# JAX distributed initialization for GPU-bearing ranks only.
# ---------------------------------------------------------------------------
compute_rank = None  # None for the IO rank

if has_gpu:
    compute_comm = comm.Split(color=0, key=rank)
    compute_rank = compute_comm.Get_rank()

    # All compute ranks agree on coordinator address (hostname of compute rank 0).
    if compute_rank == 0:
        coordinator_host = socket.gethostname()
    else:
        coordinator_host = None
    coordinator_host = compute_comm.bcast(coordinator_host, root=0)

    coordinator_address = f"{coordinator_host}:29600"
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_calculate_processes,
        process_id=compute_rank,
        local_device_ids=[0],
    )
else:
    # Non-GPU ranks must participate in comm.Split (MPI collective), but do not
    # join JAX distributed initialization.
    _ = comm.Split(color=1, key=rank)


# ---------------------------------------------------------------------------
# Simple MLP: input (batch, 256, 31) -> output (batch, 256, 31)
# 30 levels as channels + 1 horizon channel, Dense applied over last dim.
# ---------------------------------------------------------------------------
MAX_FORECAST_HORIZON = 10

_train_state = None  # lazily initialised on first training step
_mesh = None
_sample_sharding = None
_current_step = 0
_pending_example = None  # ForecastExample | None — the single in-flight forecast
_horizon_rng_key = jax.random.PRNGKey(0)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: (batch, 256, 31)
        x = nn.Dense(512)(x)  # -> (batch, 256, 512)
        x = nn.relu(x)
        x = nn.Dense(31)(x)  # -> (batch, 256, 31)
        return x


def _init_train_state(mesh):
    model = MLP()
    replicated = NamedSharding(mesh, P())  # params replicated on all GPUs
    rng = jax.random.PRNGKey(0)
    params = jax.jit(model.init, out_shardings=replicated)(
        rng, jnp.ones((1, 256, 31), dtype=jnp.float32)
    )
    tx = optax.adam(1e-3)
    return flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        pred = state.apply_fn(params, x)
        return jnp.mean((pred - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def predict_step(state, x):
    return MLP().apply(state.params, x)


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

    # this is to save the prediction to icon, but writing a JAX sharded array back to the ICON buffer (with halo re-insertion) is non-trivial,
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
    # arr_no_halo: (n_interior, nlev) -> (batch, sample_size, nlev)
    nlev = arr_no_halo.shape[1]
    n_batch = arr_no_halo.shape[0] // sample_size
    arr_samples = arr_no_halo[: n_batch * sample_size].reshape(
        n_batch, sample_size, nlev
    )
    return arr_samples


def global_jax_array(arr: xp.ndarray, sharding) -> jax.Array:
    local_data = jdlpack.from_dlpack(arr)  # JAX array on CUDA:0
    local_data = local_data.astype(jnp.float32)

    # Each compute rank contributes its local shard; JAX assembles the global array.
    array_global = jax.make_array_from_process_local_data(sharding, local_data)

    return array_global



@dataclasses.dataclass
class ForecastExample:
    """A single in-flight forecast linking a past ua snapshot to a future step.

    Attributes
    ----------
    previous_pred : jax.Array, shape (batch_global, 256, nlev+1)
        Model prediction y_t = f_θ(encode(x_t, h)) at creation time, with a
        normalised horizon channel appended (value = horizon / MAX_FORECAST_HORIZON).
        Stored as a sharded JAX array (not CuPy).
    horizon : int
        Number of ICON steps between the input snapshot and the target.
    due_step : int
        Absolute step at which the matching ground-truth ua is available
        (= creation_step + horizon).
    """

    previous_pred: jax.Array  # jnp ndarray on device
    horizon: int
    due_step: int


def _get_mesh_and_sharding():
    global _mesh, _sample_sharding

    if _mesh is None:
        _mesh = jax.make_mesh((num_calculate_processes,), ("gpu",))
        _sample_sharding = NamedSharding(_mesh, P("gpu"))

    return _mesh, _sample_sharding


def _sample_horizon() -> int:
    """Sample a random forecast horizon in [1, MAX_FORECAST_HORIZON]."""
    global _horizon_rng_key
    _horizon_rng_key, subkey = jax.random.split(_horizon_rng_key)
    # jax.random.randint upper bound is exclusive, so use MAX_FORECAST_HORIZON + 1
    # to include MAX_FORECAST_HORIZON itself in the range.
    return int(
        jax.random.randint(subkey, shape=(), minval=1, maxval=MAX_FORECAST_HORIZON + 1)
    )


def _encode_horizon_channel(arr: jax.Array, horizon: int) -> jax.Array:
    """Append a normalised horizon channel to a sharded JAX array.

    Parameters
    ----------
    arr : jax.Array, shape (..., nlev)   — must already be a sharded JAX array
    horizon : int in [0, MAX_FORECAST_HORIZON]

    Returns
    -------
    jax.Array, shape (..., nlev+1)
    """
    horizon_value = jnp.float32(horizon / MAX_FORECAST_HORIZON)
    horizon_channel = jnp.full(arr.shape[:-1] + (1,), horizon_value, dtype=jnp.float32)
    return jnp.concatenate((arr, horizon_channel), axis=-1)


# ---------------------------------------------------------------------------
# training callbacks
# ---------------------------------------------------------------------------


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training():
    """Online training callback, called by ICON at every time step.

    Four stages, depending on the time step:
    WAITING         – Between t and t+h,a prediction is in flight but not yet due;
                    return immediately with zero GPU overhead.
    INITIALIZING    – At step 0, no prediction is in flight; initialise the model
                    later a prediction is made using the initialized parameters.
    TRAINING        – When the current step matches the due step (t+h),
                    train on the pair (previous_pred → current ua)
    PREDICTION      – After training (or initialisation at step 0), predict a new
                    ua snapshot to be used as the "previous_pred" for a future step,

    Example with h=3 at step 0, h=5 at step 3:
    step 0  → INITIALIZING + PREDICTION: train (none), predict (y_0), enqueue (horizon=3, due_step=0+3=3)
    step 1  → WAITING
    step 2  → WAITING
    step 3  → TRAINING + PREDICTION: train on (y_0, x_3), PREDICTION: predict (y_3), enqueue (horizon=5, due_step=3+5=8)
        ...

    """
    if not has_gpu:
        return

    global _current_step, _pending_example, _train_state

    current_step = _current_step
    _current_step += 1


    # ---- WAITING: between t and t+h ---------------------
    if _pending_example is not None and _pending_example.due_step != current_step:
        comin.print_info(
            f"[rank={rank}] step={current_step}, waiting "
            f"(due at step={_pending_example.due_step}, horizon={_pending_example.horizon})"
        )
        return

    # ---- data preparation --------
    # only prepare data when needed
    mesh, sharding = _get_mesh_and_sharding()
    # from ICON to cupy
    ua_current_xp = sample_data(ua).astype(xp.float32)
    comin.print_info(
        f"[rank={rank}] step={current_step}, ua_current shape={ua_current_xp.shape}"
    )
    # from cupy to jax sharded array across GPUs
    ua_current = global_jax_array(ua_current_xp, sharding)


    # ---- INITIALIZING: step 0, (no training) --
    if _pending_example is None:
        if _train_state is None:
            _train_state = _init_train_state(mesh)
            comin.print_info(f"[rank={rank}] MLP initialised")


    # ---- TRAINING: train on (previous model prediction -> encode(ua_current, h)) --
    else:
        x_global = (
            _pending_example.previous_pred
        )  # prediction from previous step (y_t) (31ch)
        y_global = _encode_horizon_channel(
            ua_current, _pending_example.horizon
        )  # current ua as ground truth (x_{t+h}) (31ch)
        _train_state, loss = train_step(_train_state, x_global, y_global)
        comin.print_info(
            f"[rank={rank}] step={current_step}, "
            f"trained horizon={_pending_example.horizon}, loss={loss:.6f}"
        )


    # ---- PREDICTION: predict y_t = f(x_t) and store it -----------
    # y_t means the prediction made "at" model step t, and made "for" model step t+h.
    horizon = _sample_horizon()
    x_enc = _encode_horizon_channel(ua_current, horizon)  # 31ch input
    ua_predict = predict_step(_train_state, x_enc)  # 31ch prediction

    _pending_example = ForecastExample(
        previous_pred=ua_predict,
        horizon=horizon,
        due_step=current_step + horizon,
    )
    comin.print_info(
        f"[rank={rank}] step={current_step}, "
        f"enqueued horizon={horizon}, due_step={_pending_example.due_step}"
    )
