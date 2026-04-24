# the namelist variable "comin_process_id" should be added


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


jg = 1  # set the domain id

# request to register the variable
var_descriptor = ("comin_process_id", jg)
comin.var_request_add(var_descriptor, lmodexclusive=False)
comin.metadata_set(var_descriptor, zaxis_id=comin.COMIN_ZAXIS_2D)


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global comin_process_id
    comin_process_id = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        ("comin_process_id", jg),
        comin.COMIN_FLAG_WRITE | DEVICE_SYNC_FLAG,
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def simple_python_callbackfct():
    # print("simple_python_callbackfct called!", file=sys.stderr)
    comin_process_id_np = xp.asarray(comin_process_id)
    rank_value = comin.parallel_get_host_mpi_rank()
    if DEVICE_SYNC_FLAG != 0:
        rank_value += 100
    comin_process_id_np[:] = rank_value

'''
# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import numpy as np

# %%
ds_ll = xr.open_dataset(
    "/work/mh0033/m300883/Project_week_global/MEssE/build_gpu/messe_env/build_dir/icon-model/experiments/atm_tracer_Hadley_comin_portability/native_ml_20080901T000000Z.nc"
)
# %%

plotvar = "comin_process_id"
# %matplotlib widget

cx = ds_ll["clon"][:]
cy = ds_ll["clat"][:]

data = ds_ll[plotvar].isel(time=0).values
unique_vals = np.unique(data[~np.isnan(data)]).astype(int)
n = len(unique_vals)

cmap = plt.cm.get_cmap("tab20", n)
bounds = np.arange(n + 1) - 0.5
norm = plt.matplotlib.colors.BoundaryNorm(bounds, ncolors=n)
# remap data values to indices 0..n-1
val_to_idx = {v: i for i, v in enumerate(unique_vals)}
data_idx = np.vectorize(val_to_idx.get)(data.astype(int))

fig, axs = plt.subplots(
    1, 1, figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()}
)
axs.add_feature(cartopy.feature.BORDERS, edgecolor="k", zorder=101)
axs.coastlines(zorder=101)
sc = axs.scatter(
    cx,
    cy,
    c=data_idx,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    s=10,
    zorder=1,
)
cbar = plt.colorbar(sc, ax=axs, orientation="horizontal", pad=0.05)
cbar.set_ticks(np.arange(n))
cbar.set_ticklabels([str(v) for v in unique_vals])
cbar.set_label("MPI rank (GPU ranks = rank + 100)")


'''