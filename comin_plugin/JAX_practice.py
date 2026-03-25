# %%
import comin

# %%
import sys
import os

import numpy as np
import json
import glob
import math
from datetime import datetime
import getpass

import mpi4py.MPI as MPI

# %%
import torch
from torch.utils.data import DataLoader
from torch.utils import dlpack as torch_dlpack

# %%
import functools
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils

PyTree = Any
Metrics = Dict[str, Tuple[jax.Array, ...]]


# %%
# Check for CUDA-aware MPI support
try:
    from mpi4py.util import dtlib

    CUDA_AWARE_MPI = True
except ImportError:
    CUDA_AWARE_MPI = False

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
proc0_shift = 1  # Use first rank for gathering data, keep the same with run_icon_gpu.sh


# %%
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


# %%
# utility to extract non-halo data and global indices
def no_halo_data(data_array):
    """Extracts non-halo data and corresponding global indices."""
    nc = domain.cells.ncells
    global_idx = np.asarray(domain.cells.glb_index, dtype=np.int64) - 1

    # Convert input to numpy and flatten
    data_np = np.asarray(data_array).ravel("F")[:nc]
    decomp_np = np.asarray(domain_np).ravel("F")[:nc]

    # Mask for interior cells (where domain == 0)
    halo_mask = decomp_np == 0

    return data_np[halo_mask], global_idx[:nc][halo_mask]


# %%


# Gather all ranks to root=0 with optional CUDA-aware MPI path.
def util_gather(
    data_array,
    cpu_rank,
    cuda_aware_mpi,
    root=0,
):
    vals, idxs = no_halo_data(data_array)
    n_local = int(vals.size)
    counts = comm.gather(n_local, root=root)

    use_cuda_buffers = bool(cuda_aware_mpi and torch.cuda.is_available())

    recv_vals = None
    displs = None

    if cpu_rank == root:
        counts = np.asarray(counts, dtype=np.int32)
        displs = np.zeros(len(counts), dtype=np.int32)
        displs[1:] = np.cumsum(counts[:-1])
        n_total = int(np.sum(counts))

        if use_cuda_buffers:
            recv_vals = torch.empty(n_total, dtype=torch.float32, device="cuda")
        else:
            recv_vals = np.empty(n_total, dtype=np.float32)

    if use_cuda_buffers:
        send_vals = torch.as_tensor(vals, dtype=torch.float32, device="cuda")
        comm.Gatherv(send_vals, [recv_vals, counts, displs, MPI.FLOAT], root=root)
    else:
        send_vals = vals.astype(np.float32, copy=False)
        comm.Gatherv(send_vals, [recv_vals, counts, displs, MPI.FLOAT], root=root)

    if cpu_rank != root:
        return None

    if use_cuda_buffers:
        vals_jax = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(recv_vals))
    else:
        vals_jax = jnp.asarray(recv_vals)

    return vals_jax


# %%


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def data_test_callback():
    global tas, sfcwind, domain

    batch_size = 512
    input_dim = 16
    ouput_dim = 16

    local_prognostic_cells = int(np.sum(np.asarray(domain_np).ravel("F")[:nc] == 0))
    rank_role = "IO" if local_prognostic_cells == 0 else "COMPUTE"
    print(
        f"[rank={cpu_rank}] role={rank_role} local_prognostic_cells={local_prognostic_cells}",
        file=sys.stderr,
    )

    gather_input = util_gather(tas, cpu_rank, CUDA_AWARE_MPI, root=0)
    gather_output = util_gather(sfcwind, cpu_rank, CUDA_AWARE_MPI, root=0)

    if gather_input is None or gather_output is None:
        return

    if cpu_rank == 0:

        # info about devices and MPI
        print(f"CUDA-aware MPI: {CUDA_AWARE_MPI}")
        local_devices = jax.local_devices()
        print(f"Rank 0 local JAX devices: {local_devices}")

        # prepare data
        size_input = int(gather_input.shape[0])
        size_output = int(gather_output.shape[0])

        # reshape the gathered data to (num_samples, feature_dim) for dataloader
        n_input = (size_input // input_dim) * input_dim
        n_output = (size_output // ouput_dim) * ouput_dim

        gather_input = gather_input[:n_input].reshape(-1, input_dim)
        gather_output = gather_output[:n_output].reshape(-1, ouput_dim)

        print(
            f"Gathered input shape: {gather_input.shape}, Gathered output shape: {gather_output.shape}"
        )

        # prepare a 4-GPU mesh on rank 0
        mesh = Mesh(np.array(jax.devices()), ("data",))

        sharded_input = jax.device_put(gather_input, NamedSharding(mesh, P("data")))
        sharded_output = jax.device_put(gather_output, NamedSharding(mesh, P("data")))

        print(
            f"Sharded input shape: {sharded_input.shape}, Sharded output shape: {sharded_output.shape}"
        )
        for shard in sharded_input.addressable_shards:
            print(
                f"Input shard index={shard.index} device={shard.device} shape={shard.data.shape}",
                file=sys.stderr,
            )
        for shard in sharded_output.addressable_shards:
            print(
                f"Output shard index={shard.index} device={shard.device} shape={shard.data.shape}",
                file=sys.stderr,
            )
