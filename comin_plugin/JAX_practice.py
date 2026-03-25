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
proc0_shift = 4  # Use first 4 ranks for IO, adjust as the same as run_icon_gpu.sh


IO_RANKS = [0, 1, 2, 3]
FIRST_COMPUTE_RANK = 4
GROUP_SIZE = 3


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
# rank grouping and owner mapping
# 0<-[4,5,6], 1<-[7,8,9], 2<-[10,11,12], 3<-[13,14,15]


def build_io_groups(
    world_size,
    io_ranks=IO_RANKS,
    first_compute=FIRST_COMPUTE_RANK,
    group_size=GROUP_SIZE,
):
    groups = {r: [] for r in io_ranks}
    compute_ranks = list(range(first_compute, world_size))

    for i, r in enumerate(compute_ranks):
        owner_idx = i // group_size
        if owner_idx >= len(io_ranks):
            # strict mode: if more compute groups than IO roots, stop or raise
            # raise RuntimeError("Not enough IO ranks for compute groups")
            break
        groups[io_ranks[owner_idx]].append(r)

    owner_of = {}
    for io_root, members in groups.items():
        owner_of[io_root] = io_root
        for r in members:
            owner_of[r] = io_root
    return groups, owner_of


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


# grouped gather with CUDA-aware MPI path (per subgroup only)
def util_gather_grouped(
    data_array,
    cpu_rank,  # 0-15
    cuda_aware_mpi,
    io_ranks=IO_RANKS,
    first_compute=FIRST_COMPUTE_RANK,
    group_size=GROUP_SIZE,
):
    world_size = comm.Get_size()
    groups, owner_of = build_io_groups(world_size, io_ranks, first_compute, group_size)

    if cpu_rank not in owner_of:
        return None

    io_root = owner_of[cpu_rank]
    subgroup_members = [io_root] + groups[io_root]

    color = io_root if cpu_rank in subgroup_members else MPI.UNDEFINED
    subcomm = comm.Split(color=color, key=cpu_rank)

    if subcomm == MPI.COMM_NULL:
        return None

    try:
        sub_rank = subcomm.Get_rank()

        # local payload: remove halo and keep global index
        vals, idxs = no_halo_data(data_array)

        # gather sizes first
        n_local = np.array([vals.size], dtype=np.int32)
        counts = subcomm.gather(int(n_local[0]), root=0)

        recv_vals = None
        recv_idxs = None
        displs = None

        if sub_rank == 0:
            displs = np.zeros(len(counts), dtype=np.int32)
            displs[1:] = np.cumsum(np.array(counts[:-1], dtype=np.int32))
            n_total = int(np.sum(counts))

            if cuda_aware_mpi:
                dev = torch.device("cuda")
                recv_vals = torch.empty(n_total, dtype=torch.float32, device=dev)
                recv_idxs = torch.empty(n_total, dtype=torch.int64, device=dev)
            else:
                recv_vals = np.empty(n_total, dtype=np.float32)
                recv_idxs = np.empty(n_total, dtype=np.int64)

        if cuda_aware_mpi:
            send_vals = torch.as_tensor(vals, device="cuda", dtype=torch.float32)
            send_idxs = torch.as_tensor(idxs, device="cuda", dtype=torch.int64)
            subcomm.Gatherv(send_vals, [recv_vals, counts, displs, MPI.FLOAT], root=0)
            subcomm.Gatherv(send_idxs, [recv_idxs, counts, displs, MPI.INT64_T], root=0)
        else:
            subcomm.Gatherv(vals, [recv_vals, counts, displs, MPI.FLOAT], root=0)
            subcomm.Gatherv(idxs, [recv_idxs, counts, displs, MPI.INT64_T], root=0)

        if cpu_rank != io_root:
            return None

        # on each IO root: convert to JAX array
        if cuda_aware_mpi:
            vals_jax = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(recv_vals))
            idxs_jax = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(recv_idxs))
        else:
            vals_jax = jnp.asarray(recv_vals)
            idxs_jax = jnp.asarray(recv_idxs)

        return {
            "io_root": io_root,
            "group_members": subgroup_members,
            "values": vals_jax,
            "indices": idxs_jax,
        }
    finally:
        # Crucial: Free the sub-communicator to avoid memory leaks
        if subcomm != MPI.COMM_NULL:
            subcomm.Free()


def build_io_mesh(io_ranks=IO_RANKS):
    # requires distributed JAX init done for all ranks
    io_devices = [d for d in jax.devices() if d.process_index in io_ranks]
    if len(io_devices) != len(io_ranks):
        raise RuntimeError(
            f"Expected {len(io_ranks)} IO devices, got {len(io_devices)}"
        )
    mesh = Mesh(np.array(io_devices), axis_names=("io",))
    return mesh


def make_global_io_array(local_batch, mesh):
    # local_batch lives on each IO rank as one shard
    # local shape: [B, F]
    # global shape: [4, B, F] with axis 0 sharded by io
    pspec = P("io", None, None)
    global_arr = multihost_utils.host_local_array_to_global_array(
        local_batch[None, ...], mesh, pspec
    )
    return global_arr


@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def data_test_callback():
    global tas, sfcwind, domain
    world_size = comm.Get_size()
    groups, owner_of = build_io_groups(world_size)

    role = "IO" if cpu_rank in IO_RANKS else "COMPUTE"
    owner = owner_of.get(cpu_rank, None)
    local_cells = int(np.sum(domain_np == 0))
    print(
        f"[rank={cpu_rank}] role={role} owner_io={owner} local_nonhalo_cells={local_cells} "
        f"cuda_aware_mpi={CUDA_AWARE_MPI}",
        file=sys.stderr,
    )

    if cpu_rank in IO_RANKS:
        print(
            f"[rank={cpu_rank}] subgroup_members={ [cpu_rank] + groups.get(cpu_rank, []) }",
            file=sys.stderr,
        )

    gather_out = util_gather_grouped(tas, cpu_rank, CUDA_AWARE_MPI)

    if gather_out is None:
        return

    vals = gather_out["values"]
    idxs = gather_out["indices"]
    print(
        f"[rank={cpu_rank}] gathered_shard: values_shape={tuple(vals.shape)} values_dtype={vals.dtype} "
        f"indices_shape={tuple(idxs.shape)} indices_dtype={idxs.dtype}",
        file=sys.stderr,
    )

    try:
        mesh = build_io_mesh()
        print(
            f"[rank={cpu_rank}] mesh_shape={mesh.devices.shape} mesh_axis={mesh.axis_names}",
            file=sys.stderr,
        )

        local_batch = jnp.asarray(vals).reshape(-1, 1)
        global_arr = make_global_io_array(local_batch, mesh)
        print(
            f"[rank={cpu_rank}] global_sharded_shape={global_arr.shape} global_sharding={global_arr.sharding}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[rank={cpu_rank}] mesh_test_skipped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
