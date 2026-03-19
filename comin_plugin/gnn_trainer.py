"""
This is a ComIn Python plugin project week 2026.
"""

# %%
import comin
import sys
import os
import shutil

# %%
import numpy as np
import numpy.ma as ma
import json
import glob
import math
from datetime import datetime

# %%
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


import getpass

# Check for CUDA-aware MPI support
try:
    from mpi4py.util import dtlib

    CUDA_AWARE_MPI = True
except ImportError:
    CUDA_AWARE_MPI = False

# Import model and utilities
from gnn_model import SimpleGNN
from utils import (
    build_knn_graph_batch_numpy,
    RegionalSampleDataset,
    collate_samples,
    batch_graphs,
)

user = getpass.getuser()
torch.manual_seed(0)

# Debugging: Print number of domains
glob_data = comin.descrdata_get_global()
n_dom = glob_data.n_dom
n_dom = np.array(n_dom)
# print("number of domains:", n_dom, file=sys.stderr)


# domain info
jg = 1  # set the domain id
domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)
nc = domain.cells.ncells

## GPU group setup for DDP (if needed in the future)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)




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


# help function to collect the data from all processes

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()


def util_gather(data_array: np.ndarray, root=0, device=None):

    # 0-shifted global indices for all local cells (including halo cells):
    global_idx = np.asarray(domain.cells.glb_index) - 1

    # no. of local cells (incl. halos):
    nc = domain.cells.ncells

    # to remove empty cells
    data_array_1d = data_array.ravel("F")[0:nc]
    decomp_domain_np_1d = domain_np.ravel("F")[0:nc]
    halo_mask = decomp_domain_np_1d == 0

    # to remove halo cells
    data_array_1d = data_array_1d[halo_mask]
    global_idx = global_idx[halo_mask]

    # CUDA-aware MPI: If device is specified, use GPU tensors directly
    if device is not None and CUDA_AWARE_MPI and torch.cuda.is_available():
        # Convert to GPU tensor for CUDA-aware MPI
        data_tensor = torch.from_numpy(data_array_1d).to(device, non_blocking=True)
        global_idx_tensor = torch.from_numpy(global_idx).to(device, non_blocking=True)

        # Gather on GPU
        data_buf = comm.gather(
            (data_tensor.cpu().numpy(), global_idx_tensor.cpu().numpy()), root=root
        )
    else:
        # Standard CPU-based gather
        data_buf = comm.gather((data_array_1d, global_idx), root=root)

    # reorder received data according to global_idx
    if rank == root:
        nglobal = sum([len(gi) for _, gi in data_buf])
        if device is not None and torch.cuda.is_available():
            # Return as GPU tensor for faster downstream processing
            global_array = torch.zeros(nglobal, dtype=torch.float32, device=device)
            for data_array_i, global_idx_i in data_buf:
                idx_tensor = torch.from_numpy(global_idx_i).long().to(device)
                val_tensor = torch.from_numpy(data_array_i).float().to(device)
                global_array[idx_tensor] = val_tensor
            return global_array
        else:
            # Return as numpy array
            global_array = np.zeros(nglobal, dtype=np.float64)
            for data_array_i, global_idx_i in data_buf:
                global_array[global_idx_i] = data_array_i
            return global_array
    else:
        return None


# ============================================
# DDP Multi-GPU Configuration & Helper Functions
# ============================================

# Multi-GPU + Multi-CPU rank mapping
NUM_GPUS = 4
CPU_RANKS_PER_GPU = 4
NUM_CPU_RANKS = 16


def get_cpu_ranks_for_gpu(gpu_rank):
    """
    Map GPU rank to assigned CPU ranks (round-robin).
    GPU 0 <- CPU [0, 1, 2, 3]
    GPU 1 <- CPU [4, 5, 6, 7]
    GPU 2 <- CPU [8, 9, 10, 11]
    GPU 3 <- CPU [12, 13, 14, 15]
    """
    start_cpu_rank = gpu_rank * CPU_RANKS_PER_GPU
    return list(range(start_cpu_rank, start_cpu_rank + CPU_RANKS_PER_GPU))


def util_gather_subset(data_array: np.ndarray, cpu_ranks, device=None, comm=None):
    """
    Gather data from only specified CPU ranks (subset gather).
    
    Args:
        data_array: local data to contribute
        cpu_ranks: list of CPU ranks to gather from (e.g., [0, 1, 2, 3])
        device: torch device (GPU/CPU) for output
        comm: MPI communicator (uses global if None)
    
    Returns:
        Gathered data (tensor or numpy array) ordered by global index, or None if not root
    """
    if comm is None:
        comm = globals()['comm']  # Use global comm if not provided
    
    # Get local rank from global comm
    local_rank = comm.Get_rank()
    
    # 0-shifted global indices for all local cells (including halo cells):
    global_idx = np.asarray(domain.cells.glb_index) - 1
    nc = domain.cells.ncells
    
    # Remove empty cells
    data_array_1d = data_array.ravel("F")[0:nc]
    decomp_domain_np_1d = domain_np.ravel("F")[0:nc]
    halo_mask = decomp_domain_np_1d == 0
    
    # Remove halo cells
    data_array_1d = data_array_1d[halo_mask]
    global_idx = global_idx[halo_mask]
    
    # Prepare data to send
    if local_rank in cpu_ranks:
        local_data = (data_array_1d, global_idx)
    else:
        local_data = None  # Non-participating rank sends None
    
    # Gather to GPU rank 0 (in the subset group)
    root_cpu_rank = cpu_ranks[0]
    
    if CUDA_AWARE_MPI and device is not None and torch.cuda.is_available():
        # CUDA-aware MPI path
        if local_rank in cpu_ranks:
            data_tensor = torch.from_numpy(data_array_1d).to(device, non_blocking=True)
            global_idx_tensor = torch.from_numpy(global_idx).to(device, non_blocking=True)
            local_data = (data_tensor.cpu().numpy(), global_idx_tensor.cpu().numpy())
    
    data_buf = comm.gather(local_data, root=root_cpu_rank)
    
    # Reorder received data according to global_idx (only root receives non-None)
    if local_rank == root_cpu_rank and data_buf is not None:
        # Filter out None entries from non-participating ranks
        data_buf = [d for d in data_buf if d is not None]
        nglobal = sum([len(gi) for _, gi in data_buf])
        
        if device is not None and torch.cuda.is_available():
            # Return as GPU tensor
            global_array = torch.zeros(nglobal, dtype=torch.float32, device=device)
            for data_array_i, global_idx_i in data_buf:
                idx_tensor = torch.from_numpy(global_idx_i).long().to(device)
                val_tensor = torch.from_numpy(data_array_i).float().to(device)
                global_array[idx_tensor] = val_tensor
            return global_array
        else:
            # Return as numpy array
            global_array = np.zeros(nglobal, dtype=np.float64)
            for data_array_i, global_idx_i in data_buf:
                global_array[global_idx_i] = data_array_i
            return global_array
    else:
        return None


def prepare_data(gpu_rank, device, timing_info):
    """
    Prepare training data for a specific GPU rank.
    Each GPU gathers data from its assigned CPU ranks.
    
    Args:
        gpu_rank: GPU rank (0-3)
        device: torch device
        timing_info: dict with timing metadata (start_time_str, current_time_str, etc.)
    
    Returns:
        dict with keys: x_full, y_full, pos_np, num_nodes
    """
    cpu_ranks = get_cpu_ranks_for_gpu(gpu_rank)
    
    # Gather data from assigned CPU ranks
    tas_np_glb = util_gather_subset(np.asarray(tas), cpu_ranks, device=device)
    sfcwind_np_glb = util_gather_subset(np.asarray(sfcwind), cpu_ranks, device=device)
    
    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather_subset(cx, cpu_ranks, device=None)  # Coordinates on CPU
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather_subset(cy, cpu_ranks, device=None)
    
    # Process data
    if torch.is_tensor(tas_np_glb):
        num_nodes = tas_np_glb.shape[0]
        x_full = tas_np_glb.unsqueeze(1).float()  # [num_nodes, 1]
        y_full = sfcwind_np_glb.unsqueeze(1).float()  # [num_nodes, 1]
        pos_np = np.column_stack([cx_glb, cy_glb])
    else:
        num_nodes = tas_np_glb.shape[0]
        x_full = torch.FloatTensor(tas_np_glb).unsqueeze(1)  # [num_nodes, 1]
        y_full = torch.FloatTensor(sfcwind_np_glb).unsqueeze(1)  # [num_nodes, 1]
        pos_np = np.column_stack([cx_glb, cy_glb])
    
    return {
        "x_full": x_full,
        "y_full": y_full,
        "pos_np": pos_np,
        "num_nodes": num_nodes,
    }


def train_worker(gpu_rank, world_size, timing_info, scratch_dir):
    """
    DDP training worker for a single GPU rank.
    
    Args:
        gpu_rank: GPU rank (0-3)
        world_size: Total number of GPU workers (4)
        timing_info: dict with timing metadata
        scratch_dir: directory for checkpoints and logs
    """
    try:
        # ============================================
        #      DDP Setup
        # ===========================================
        ddp_setup(gpu_rank, world_size)
        device = torch.device(f"cuda:{gpu_rank}")
        
        if gpu_rank == 0:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"🚀 DDP Training initialized with {world_size} GPU(s)", file=sys.stderr)
            print(f"   GPU 0 <- CPU [0, 1, 2, 3]", file=sys.stderr)
            print(f"   GPU 1 <- CPU [4, 5, 6, 7]", file=sys.stderr)
            print(f"   GPU 2 <- CPU [8, 9, 10, 11]", file=sys.stderr)
            print(f"   GPU 3 <- CPU [12, 13, 14, 15]", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
        
        # ============================================
        #      Data Preparation
        # ===========================================
        cpu_ranks = get_cpu_ranks_for_gpu(gpu_rank)
        if gpu_rank == 0:
            print(f"GPU {gpu_rank} gathering from CPU ranks {cpu_ranks}", file=sys.stderr)
        
        data = prepare_data(gpu_rank, device, timing_info)
        x_full = data["x_full"]
        y_full = data["y_full"]
        pos_np = data["pos_np"]
        num_nodes = data["num_nodes"]
        
        if gpu_rank == 0:
            print(f"✓ Data prepared: {num_nodes} nodes", file=sys.stderr)
        
        # ============================================
        #      Dataset & DataLoader
        # ===========================================
        sample_size = 2500
        dataset = RegionalSampleDataset(
            x_full, y_full, pos_np, sample_size=sample_size, k=6, extended_k=8
        )
        num_samples = len(dataset)
        batch_size = max(1, math.ceil(num_samples / 2))
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_samples,
        )
        
        if gpu_rank == 0:
            print(f"✓ Dataset created: {num_samples} samples, batch_size={batch_size}", file=sys.stderr)
        
        # ============================================
        #      Model Initialization / Loading
        # ===========================================
        start_time_str = timing_info["start_time_str"]
        current_time_str = timing_info["current_time_str"]
        checkpoint_path = f"{scratch_dir}/net_{start_time_str}.pth"
        
        if os.path.exists(checkpoint_path) and gpu_rank == 0:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            net = SimpleGNN(in_channels=1, hidden_channels=32, out_channels=1, num_layers=3)
            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if gpu_rank == 0:
                print(f"✓ Model loaded from checkpoint", file=sys.stderr)
        else:
            net = SimpleGNN(in_channels=1, hidden_channels=32, out_channels=1, num_layers=3)
            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
            if gpu_rank == 0:
                print(f"✓ Model initialized (new)", file=sys.stderr)
        
        # Wrap model with DDP
        net = DDP(net, device_ids=[gpu_rank])
        
        # ============================================
        #      Training Loop
        # ===========================================
        num_batches = len(dataloader)
        num_epochs = 1
        lossfunc = torch.nn.MSELoss()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            if gpu_rank == 0:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"Epoch {epoch+1}/{num_epochs}", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)
            
            for batch_idx, (x_list, y_list, edge_list, mask_list) in enumerate(dataloader):
                x_batch, y_batch, edge_batch, mask_batch = batch_graphs(
                    x_list, y_list, edge_list, mask_list
                )
                
                x_batch = x_batch.to(device, non_blocking=False)
                y_batch = y_batch.to(device, non_blocking=False)
                edge_batch = edge_batch.to(device, non_blocking=False)
                mask_batch = mask_batch.to(device, non_blocking=False)
                
                optimizer.zero_grad()
                y_hat_extended = net(x_batch, edge_batch)
                y_hat = y_hat_extended[mask_batch]
                
                batch_loss = lossfunc(y_hat, y_batch)
                batch_loss.backward()
                optimizer.step()
                
                epoch_losses.append(batch_loss.item())
                if gpu_rank == 0:
                    print(f"  Batch {batch_idx+1}/{num_batches}: Loss = {batch_loss.item():.6e}", file=sys.stderr)
            
            epoch_avg_loss = np.mean(epoch_losses)
            losses.append(epoch_avg_loss)
            if gpu_rank == 0:
                print(f"  Epoch {epoch+1} completed: Avg Loss = {epoch_avg_loss:.6e}", file=sys.stderr)
        
        if gpu_rank == 0:
            print(f"\n✓ GNN training completed (GPU {gpu_rank})", file=sys.stderr)
            print(f"  Final epoch avg loss: {losses[-1]:.6e}", file=sys.stderr)
        
        # ============================================
        #      Save Checkpoint (GPU rank 0 only)
        # ===========================================
        if gpu_rank == 0:
            torch.save(
                {
                    "model_state_dict": net.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"✓ Model saved to {checkpoint_path}", file=sys.stderr)
            
            # Save loss logs
            with open(f"{scratch_dir}/log_{current_time_str}.txt", "w") as f:
                f.write(f"{losses[-1]}\n")
            
            with open(f"{scratch_dir}/log_detailed_{current_time_str}.txt", "w") as f:
                f.write(f"# Epoch-level average losses for this timestep\n")
                for i, loss in enumerate(losses):
                    f.write(f"Epoch {i+1}: {loss:.6e}\n")
        
        # Cleanup
        destroy_process_group()
        
    except Exception as e:
        print(f"❌ Error in GPU rank {gpu_rank}: {e}", file=sys.stderr)
        raise


# training callback
@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    """
    Main training callback. Orchestrates DDP training across 4 GPU workers.
    Each GPU worker gathers data from its assigned 4 CPU ranks.
    """

    # ============================================
    #      GPU detection
    # ===========================================
    if rank == 0:
        if torch.cuda.is_available():
            print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}", file=sys.stderr)
            print(f"   GPU count: {torch.cuda.device_count()}", file=sys.stderr)
            if CUDA_AWARE_MPI:
                print(f"   💡 CUDA-aware MPI: ENABLED", file=sys.stderr)
            else:
                print(f"   ⚠ CUDA-aware MPI: DISABLED", file=sys.stderr)
        else:
            print(f"\n💻 No GPU detected, using CPU", file=sys.stderr)

    # ============================================
    #      check time condition
    # ===========================================
    start_time_obj = comin.descrdata_get_simulation_interval().run_start
    current_time_obj = comin.current_get_datetime()

    start_time_str = str(start_time_obj).replace(" ", "_").replace(":", "-")
    current_time_str = str(current_time_obj).replace(" ", "_").replace(":", "-")

    start_time = datetime.fromisoformat(str(start_time_obj))
    current_time = datetime.fromisoformat(str(current_time_obj))

    elapsed_time = current_time - start_time
    elapsed_hours = elapsed_time.total_seconds() / 3600
    threshold_hours = 2.0
    should_train = elapsed_hours > threshold_hours

    if not should_train:
        if rank == 0:
            print(
                f"\n⏸ Waiting for training time (elapsed: {elapsed_hours:.2f}/{threshold_hours:.1f} hours)",
                file=sys.stderr,
            )
        return

    # ============================================
    #      Prepare DDP Training
    # ===========================================
    if rank == 0:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"🚀 Starting DDP Multi-GPU Training", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    scratch_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
    os.makedirs(scratch_dir, exist_ok=True)

    timing_info = {
        "start_time": start_time,
        "current_time": current_time,
        "start_time_str": start_time_str,
        "current_time_str": current_time_str,
        "elapsed_hours": elapsed_hours,
        "threshold_hours": threshold_hours,
    }

    # ============================================
    #      Spawn DDP Workers
    # ===========================================
    try:
        mp.spawn(
            train_worker,
            args=(NUM_GPUS, timing_info, scratch_dir),
            nprocs=NUM_GPUS,
            join=True,
        )
        if rank == 0:
            print(f"\n✓ DDP training completed successfully", file=sys.stderr)
    except Exception as e:
        if rank == 0:
            print(f"❌ DDP training failed: {e}", file=sys.stderr)
        raise

    # ============================================
    #      Post-Training (Main rank 0 only)
    # ===========================================
    if rank == 0:
        # Write monitoring status JSON
        try:
            status_data = {
                "timestamp": current_time.isoformat(),
                "simulation": {
                    "start_time": str(start_time),
                    "current_time": str(current_time),
                    "elapsed_time": str(elapsed_time),
                    "n_domains": int(n_dom) if n_dom is not None else 1,
                    "training_method": "DDP (4 GPUs)",
                },
                "hardware": {
                    "gpu_available": torch.cuda.is_available(),
                    "cuda_aware_mpi": CUDA_AWARE_MPI and torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                },
                "training": {
                    "model_type": "GNN (Sample-based, DDP)",
                    "num_gpus": NUM_GPUS,
                    "training_enabled": should_train,
                    "elapsed_hours": elapsed_hours,
                },
            }

            with open(f"{scratch_dir}/monitor_status.json", "w") as f:
                json.dump(status_data, f, indent=2)
            print(f"✓ Monitoring status written", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not write monitoring status: {e}", file=sys.stderr)
