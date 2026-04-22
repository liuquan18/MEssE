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
        flag=comin.COMIN_FLAG_WRITE,
    )


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def simple_python_callbackfct():
    # print("simple_python_callbackfct called!", file=sys.stderr)
    comin_process_id_np = xp.asarray(comin_process_id)
    rank_value = comin.parallel_get_host_mpi_rank()
    if DEVICE_SYNC_FLAG != 0:
        rank_value += 100
    comin_process_id_np[:] = rank_value
