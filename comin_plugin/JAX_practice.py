# %%
import comin

#%%
import sys
import os

import numpy as np
import json
import glob
import math
from datetime import datetime
import getpass

import mpi4py.MPI as MPI
#%%
import torch
from torch.utils.data import DataLoader

# %%
import functools
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]


# %%
# Check for CUDA-aware MPI support
try:
    from mpi4py.util import dtlib

    CUDA_AWARE_MPI = True
except ImportError:
    CUDA_AWARE_MPI = False

# %%
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
cpu_rank = comm.Get_rank()


# %%
user = getpass.getuser()

# %%
# domain info
jg = 1  # set the domain id
domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)
nc = domain.cells.ncells
#%%

## secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global tas, sfcwind, domain
    # Read existing ICON variables directly (no need for descriptors since they're not registered)
    tas = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", jg), flag=comin.COMIN_FLAG_READ
    )
    sfcwind = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("sfcwind", jg), flag=comin.COMIN_FLAG_READ
    )




jg = 1  # set the domain id

def util_gather(data_array, root=0):
    # Convert ComIn variable wrapper to NumPy array
    data_np = np.asarray(data_array)

    global_idx = np.asarray(domain.cells.glb_index, dtype=np.int64) - 1
    nc = domain.cells.ncells

    data_array_1d = data_np.ravel("F")[:nc]
    decomp_domain_np_1d = domain_np.ravel("F")[:nc]
    halo_mask = (decomp_domain_np_1d == 0)

    data_array_1d = data_array_1d[halo_mask]
    global_idx = global_idx[:nc][halo_mask]

    data_buf = comm.gather((data_array_1d, global_idx), root=root)

    if cpu_rank == root:
        nglobal = sum(len(gi) for _, gi in data_buf)
        global_array = np.zeros(nglobal, dtype=np.float64)
        for data_array_i, global_idx_i in data_buf:
            global_array[global_idx_i] = data_array_i
        return global_array
    return None
    

comm   = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank   = comm.Get_rank()

@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def train_call_back():
    global tas, sfcwind, domain
    num_prognostic_cells = np.sum(domain_np == 0)

    tas = util_gather(tas)


    if cpu_rank == 0:
        if num_prognostic_cells == 0:
            print("=" * 50, )
            print("Rank 0 is an infrastructure/IO rank and is NOT taking part in calculations.", file=sys.stderr)
            print(f"Rank {rank}: tas shape after gather: {tas.shape}", file=sys.stderr)

        else:
            print(f"Rank 0 is participating in calculations with {num_prognostic_cells} cells.", file=sys.stderr)
