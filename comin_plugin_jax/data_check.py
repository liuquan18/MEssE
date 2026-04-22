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
_plot_step = 0  # step counter for ua plot file names
_pending_example = None  # ForecastExample | None — the single in-flight forecast
_horizon_rng_key = jax.random.PRNGKey(0)


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


def global_jax_array(arr: xp.ndarray, sharding) -> jax.Array:
    local_data = jdlpack.from_dlpack(arr)  # JAX array on CUDA:0
    local_data = local_data.astype(jnp.float32)

    # Each compute rank contributes its local shard; JAX assembles the global array.
    array_global = jax.make_array_from_process_local_data(sharding, local_data)

    return array_global


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def plot_ua_first_level():
    """Plot ua (zonal wind) at level 0 — one subplot per MPI rank (all domains).

    All MPI ranks participate (including the IO/CPU rank that also holds a domain
    piece). Interior (prognostic) cells are coloured by wind speed; halo cells are
    shown in grey. Runs exactly once. Output: ./ua_level0_all_ranks_0000.png
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, safe on compute nodes
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np  # needed only for MPI payload and matplotlib (host-side only)

    global _plot_step

    # ALL ranks must share this guard — do NOT place any per-rank early return
    # before comm.gather() or the MPI collective will deadlock.
    if _plot_step > 0:
        return

    nc = domain.cells.ncells  # total local cells (interior + halo)

    # ------------------------------------------------------------------
    # Extract ua at level 0 via jnp.
    # GPU ranks (0-3): jnp wraps CUDA buffer.
    # IO rank (4):     jnp wraps CPU buffer (JAX runs on CPU without dist init).
    # ------------------------------------------------------------------
    ua_jnp = jnp.asarray(ua)
    # Squeeze trailing singleton dims: e.g. (nproma, nlev, nblk, 1, 1) → (nproma, nlev, nblk)
    while ua_jnp.ndim > 3 and ua_jnp.shape[-1] == 1:
        ua_jnp = ua_jnp[..., 0]
    # Slice level 0 → (nproma, nblk), then F-order flatten → (nc,)
    ua_cells = ua_jnp[:, 0, :].T.ravel()[:nc]

    # ------------------------------------------------------------------
    # Coordinates & domain decomposition mask — all computed with jnp
    # ------------------------------------------------------------------
    clon_cells    = jnp.rad2deg(jnp.asarray(domain.cells.clon)).T.ravel()[:nc]
    clat_cells    = jnp.rad2deg(jnp.asarray(domain.cells.clat)).T.ravel()[:nc]
    decomp_cells  = jnp.asarray(domain.cells.decomp_domain).T.ravel()[:nc]
    interior_mask = decomp_cells == 0  # True = prognostic/interior cell

    # ------------------------------------------------------------------
    # Move to host numpy for MPI gather and matplotlib
    # ------------------------------------------------------------------
    payload = {
        "ua":       np.asarray(ua_cells),
        "clon":     np.asarray(clon_cells),
        "clat":     np.asarray(clat_cells),
        "interior": np.asarray(interior_mask),
        "rank":     rank,      # global MPI rank (0..world_size-1)
        "has_gpu":  has_gpu,   # for subplot labelling
    }

    # Gather from ALL MPI ranks (comm includes the IO/CPU rank too)
    all_data = comm.gather(payload, root=0)

    if rank == 0:
        # Sort by global rank for a consistent left-to-right ordering
        all_data = sorted(all_data, key=lambda d: d["rank"])
        n = len(all_data)

        # Dynamic 2-row layout: ceil(n/2) columns
        ncols = (n + 1) // 2
        nrows = 2

        # Shared colorscale: 2nd–98th percentile across all ranks' interior ua
        all_interior_ua = np.concatenate(
            [d["ua"][d["interior"]] for d in all_data if d["interior"].any()]
        )
        vmin, vmax = np.percentile(all_interior_ua, [2, 98])

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.suptitle(
            f"ua (zonal wind) level 0 — all {n} MPI domains",
            fontsize=14,
        )

        axs_flat = axs.ravel()
        sc = None
        for ax, d in zip(axs_flat, all_data):
            interior = d["interior"]
            halo = ~interior
            label = f"rank {d['rank']} ({'GPU' if d['has_gpu'] else 'CPU/IO'})"

            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")

            # Halo cells — grey, tiny, semi-transparent (shows decomp boundary)
            if halo.any():
                ax.scatter(
                    d["clon"][halo],
                    d["clat"][halo],
                    s=0.3,
                    c="lightgray",
                    alpha=0.5,
                    transform=ccrs.PlateCarree(),
                    rasterized=True,
                    label="halo",
                )

            # Interior (prognostic) cells — coloured by ua
            if interior.any():
                sc = ax.scatter(
                    d["clon"][interior],
                    d["clat"][interior],
                    s=0.5,
                    c=d["ua"][interior],
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    rasterized=True,
                    label="interior",
                )

            ax.set_title(label, fontsize=11)
            ax.legend(loc="lower left", markerscale=8, fontsize=7, framealpha=0.6)

        # Hide unused axes if n < nrows*ncols
        for ax in axs_flat[n:]:
            ax.set_visible(False)

        # Shared horizontal colorbar
        if sc is not None:
            cbar = fig.colorbar(
                sc,
                ax=axs_flat[:n].tolist(),
                orientation="horizontal",
                pad=0.04,
                fraction=0.03,
            )
            cbar.set_label("ua [m/s]", fontsize=11)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        fname = "./ua_level0_all_ranks_0000.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        comin.print_info(f"Saved ua plot: {fname}")

    _plot_step += 1
