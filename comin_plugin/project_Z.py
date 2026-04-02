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
# Simple MLP: input (batch, 256, 30) -> output (batch, 256, 30)
# 256 = spatial cells per sample, 30 = levels as channels
# Dense is applied independently over the last (channel) dimension.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x: (batch, 256, 30)
        x = nn.Dense(512)(x)  # -> (batch, 256, 512)
        x = nn.relu(x)
        x = nn.Dense(30)(x)  # -> (batch, 256, 30)
        return x


_train_state = None  # lazily initialised on first callback


def _init_train_state(mesh):
    model = MLP()
    replicated = NamedSharding(mesh, P())  # params replicated on all GPUs
    rng = jax.random.PRNGKey(0)
    params = jax.jit(model.init, out_shardings=replicated)(rng, jnp.ones((1, 256, 30)))
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
# data constructor: setup variables and data extraction utilities
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
    comin.print_info(f"[rank={rank}] no_halo_data output shape: {arr_no_halo.shape}")
    # arr_no_halo: (n_interior, nlev) -> (batch, sample_size, nlev)
    nlev = arr_no_halo.shape[1]
    n_batch = arr_no_halo.shape[0] // sample_size
    arr_samples = arr_no_halo[: n_batch * sample_size].reshape(
        n_batch, sample_size, nlev
    )
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
