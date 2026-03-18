"""
This is a ComIn Python plugin project week 2026.
"""

# %%
import comin
import sys
import os

import numpy as np
import json
import glob
import math
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from mpi4py import MPI

import getpass

# %%
# Check for CUDA-aware MPI support
try:
    from mpi4py.util import dtlib

    CUDA_AWARE_MPI = True
except ImportError:
    CUDA_AWARE_MPI = False

# Import model and utilities
# %%
from gnn_model import SimpleGNN
from utils import (
    RegionalSampleDataset,
    collate_samples,
    batch_graphs,
)

# %%
comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
cpu_rank = comm.Get_rank()


# %%
user = getpass.getuser()
torch.manual_seed(0)

# ============================================================================
# Module-level globals for persistent state across simulation timesteps
# ============================================================================
net = None  # Lazy-initialized during first training step
optimizer = None  # Lazy-initialized during first training step
is_model_initialized = False  # Track initialization state

# domain info
jg = 1  # set the domain id
domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)
nc = domain.cells.ncells


## secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global temp, tas, sfcwind, domain
    # Read existing ICON variables directly (no need for descriptors since they're not registered)
    temp = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", jg), flag=comin.COMIN_FLAG_READ
    )
    tas = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("tas", jg), flag=comin.COMIN_FLAG_READ
    )
    sfcwind = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("sfcwind", jg), flag=comin.COMIN_FLAG_READ
    )


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
    if cpu_rank == root:
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


# ============================================================================
# Unit function: Prepare data (data loading and gathering)
# ============================================================================
def prepare_data(x_arr, y_arr, device):
    """Gather distributed data and create DataLoader for training.

    Args:
        x_arr: Input features (e.g., tas) from local rank
        y_arr: Target values (e.g., sfcwind) from local rank
        device: Device to place data on (cpu or cuda)

    Returns:
        Tuple: (dataloader, pos_np) or (None, None) on non-root ranks
    """
    # Proceed with data gathering (only when training time has arrived)
    # Use CUDA-aware MPI if available - data will be gathered directly on GPU
    tas_np_glb = util_gather(np.asarray(x_arr), device=device)
    sfcwind_np_glb = util_gather(np.asarray(y_arr), device=device)

    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather(cx, device=device)
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather(cy, device=device)

    # on process 0, proceed with training
    if cpu_rank == 0:

        # data size info
        if torch.is_tensor(tas_np_glb):
            # Data already on GPU from CUDA-aware MPI
            x_full = tas_np_glb.unsqueeze(1).float()  # [num_nodes, 1]
            y_full = sfcwind_np_glb.unsqueeze(1).float()  # [num_nodes, 1]
            # Convert coordinates to numpy for graph construction
            pos_np = np.column_stack([cx_glb, cy_glb])
        else:
            # Fallback: data is numpy array
            x_full = torch.FloatTensor(tas_np_glb).unsqueeze(1)  # [num_nodes, 1]
            y_full = torch.FloatTensor(sfcwind_np_glb).unsqueeze(1)  # [num_nodes, 1]
            pos_np = np.column_stack([cx_glb, cy_glb])

        sample_size = 2500  # Each regional sample contains 2500 nodes

        # Create dataset and dataloader
        dataset = RegionalSampleDataset(
            x_full, y_full, pos_np, sample_size=sample_size, k=6, extended_k=8
        )

        num_samples = len(dataset)
        batch_size = max(1, math.ceil(num_samples / 2))  # target 2 batches/epoch

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_samples,
        )
        return dataloader, pos_np
    else:
        return None, None


# ============================================================================
# Unit function: Check if training should occur at this timestep
# ============================================================================
def check_training_time(threshold_hours=2.0):
    """Check if enough simulation time has elapsed to trigger training.

    Args:
        threshold_hours: Minimum elapsed hours before training starts (default: 2.0)

    Returns:
        Tuple: (should_train, start_time_str, current_time_str, elapsed_hours, threshold_hours)
    """
    start_time_obj = comin.descrdata_get_simulation_interval().run_start
    current_time_obj = comin.current_get_datetime()

    # Convert to strings for safe filenames
    start_time_str = str(start_time_obj).replace(" ", "_").replace(":", "-")
    current_time_str = str(current_time_obj).replace(" ", "_").replace(":", "-")

    # Convert to datetime objects for calculations
    start_time = datetime.fromisoformat(str(start_time_obj))
    current_time = datetime.fromisoformat(str(current_time_obj))

    # Calculate elapsed time and check if threshold has been exceeded
    elapsed_time = current_time - start_time
    elapsed_hours = elapsed_time.total_seconds() / 3600
    should_train = elapsed_hours > threshold_hours

    return (
        should_train,
        start_time_str,
        current_time_str,
        elapsed_hours,
        threshold_hours,
    )


# ============================================================================
# Unit function: Load or initialize model from checkpoint
# ============================================================================
def load_or_init_model(net, optimizer, checkpoint_path, device):
    """Load model and optimizer from checkpoint, or initialize fresh.

    Args:
        net: The neural network model
        optimizer: The optimizer instance
        checkpoint_path: Path to checkpoint file
        device: Device to load onto (cpu or cuda)

    Returns:
        Tuple: (net, optimizer) - loaded or freshly initialized
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✓ Model loaded from checkpoint: {checkpoint_path}", file=sys.stderr)
    else:
        print("✓ No checkpoint found, initializing new model", file=sys.stderr)

    return net, optimizer


# ============================================================================
# Unit function: Initialize model and optimizer
# ============================================================================
def initialize_model(device, start_time_str):
    """Create and initialize a fresh GNN model and optimizer.

    Args:
        device: Device to place model on (cpu or cuda)
        start_time_str: Start time string for checkpoint naming

    Returns:
        Tuple: (net, optimizer, scratch_dir, checkpoint_path)
    """
    # Create model on device
    net = SimpleGNN(in_channels=1, hidden_channels=32, out_channels=1, num_layers=3).to(
        device
    )

    # Create optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Set up scratch directory and checkpoint path
    scratch_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
    os.makedirs(scratch_dir, exist_ok=True)
    checkpoint_path = f"{scratch_dir}/net_{start_time_str}.pth"

    # Load from checkpoint if it exists, otherwise use fresh initialization
    net, optimizer = load_or_init_model(net, optimizer, checkpoint_path, device)

    return net, optimizer, scratch_dir, checkpoint_path


# ============================================================================
# Unit function: Run training loop for a single timestep
# ============================================================================
def run_training_loop(net, optimizer, dataloader, device, num_epochs=1):
    """Run GNN training for specified number of epochs.

    Args:
        net: Neural network model
        optimizer: Optimizer instance
        dataloader: DataLoader with batch data
        device: Device to run on (cpu or cuda)
        num_epochs: Number of training epochs (default: 1)

    Returns:
        losses: List of average losses per epoch
    """
    lossfunc = torch.nn.MSELoss()
    losses = []

    print(
        f"Training: {num_epochs} epoch(s), {len(dataloader)} batch(es) per epoch",
        file=sys.stderr,
    )

    # Train over multiple epochs
    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Epoch {epoch+1}/{num_epochs}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Loop over batches (each batch contains multiple samples)
        for batch_idx, (x_list, y_list, edge_list, mask_list) in enumerate(dataloader):
            # Merge all samples in the batch into a single big graph
            x_batch, y_batch, edge_batch, mask_batch = batch_graphs(
                x_list, y_list, edge_list, mask_list
            )

            # Move data to device
            x_batch = x_batch.to(device, non_blocking=False)
            y_batch = y_batch.to(device, non_blocking=False)
            edge_batch = edge_batch.to(device, non_blocking=False)
            mask_batch = mask_batch.to(device, non_blocking=False)

            # Forward pass
            optimizer.zero_grad()
            y_hat_extended = net(x_batch, edge_batch)
            y_hat = y_hat_extended[mask_batch]

            # Backward pass
            batch_loss = lossfunc(y_hat, y_batch)
            batch_loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss.item())
            print(
                f"  Batch {batch_idx+1}/{len(dataloader)}: Loss = {batch_loss.item():.6e}",
                file=sys.stderr,
            )

        # Report epoch statistics
        epoch_avg_loss = np.mean(epoch_losses)
        losses.append(epoch_avg_loss)
        print(
            f"  Epoch {epoch+1} completed: Avg Loss = {epoch_avg_loss:.6e}",
            file=sys.stderr,
        )

    return losses


# ============================================================================
# Unit function: Save training results and monitoring status
# ============================================================================
def save_training_results(
    net,
    optimizer,
    losses,
    global_mean_tas,
    scratch_dir,
    start_time_str,
    current_time_str,
    device,
    elapsed_hours,
    threshold_hours,
    num_nodes,
    num_samples,
    sample_size,
):
    """Save model checkpoint, losses, and monitoring status.

    Args:
        net: Trained network model
        optimizer: Optimizer state
        losses: List of epoch losses
        global_mean_tas: Global mean temperature value
        scratch_dir: Directory to save outputs
        start_time_str: Simulation start time string
        current_time_str: Current simulation time string
        device: Device used for training
        elapsed_hours: Elapsed simulation hours
        threshold_hours: Training threshold hours
        num_nodes: Total number of grid nodes
        num_samples: Number of samples in dataset
        sample_size: Size of each sample
    """
    # Save global mean tas for monitoring
    with open(
        f"{scratch_dir}/global_mean_tas_{current_time_str}.txt",
        "w",
    ) as f:
        f.write(f"{global_mean_tas:.6f}\n")

    # Save model checkpoint
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{scratch_dir}/net_{start_time_str}.pth",
    )
    print("✓ Model checkpoint saved", file=sys.stderr)

    # Save loss for this timestep (simple format for quick monitoring)
    final_epoch_avg_loss = losses[-1]
    with open(
        f"{scratch_dir}/log_{current_time_str}.txt",
        "w",
    ) as f:
        f.write(f"{final_epoch_avg_loss}\n")

    # Save detailed epoch-level losses
    with open(
        f"{scratch_dir}/log_detailed_{current_time_str}.txt",
        "w",
    ) as f:
        f.write(f"# Epoch-level average losses for this timestep\n")
        for i, epoch_loss in enumerate(losses):
            f.write(f"Epoch {i+1}: {epoch_loss:.6e}\n")

    # Save comprehensive monitoring status
    try:
        status_data = {
            "timestamp": current_time_str,
            "simulation": {
                "elapsed_hours": elapsed_hours,
                "threshold_hours": threshold_hours,
                "total_points": num_nodes,
                "output_count": len(glob.glob(f"{scratch_dir}/log_*.txt")),
            },
            "hardware": {
                "device": str(device),
                "gpu_available": torch.cuda.is_available(),
                "cuda_aware_mpi": CUDA_AWARE_MPI and torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "N/A"
                ),
            },
            "training": {
                "model_type": "GNN (Sample-based)",
                "current_loss": float(losses[-1]) if losses else 0.0,
                "num_epochs": len(losses),
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "min_loss": float(np.min(losses)) if losses else 0.0,
                "max_loss": float(np.max(losses)) if losses else 0.0,
                "num_samples": num_samples,
                "sample_size": sample_size,
            },
        }

        with open(
            f"{scratch_dir}/monitor_status.json",
            "w",
        ) as f:
            json.dump(status_data, f, indent=2)

        print("✓ Monitoring status saved", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not save monitoring status: {e}", file=sys.stderr)


# ============================================================================
# Main training callback: Orchestrates the entire training process
# ============================================================================
@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def training_callback():
    """Main training orchestrator.

    Called by ICON via comin callback at each atmosphere write timestep.
    Handles: time checking, data loading, lazy model initialization,
    training loop execution, and result saving.
    """
    global net, optimizer, is_model_initialized

    # Only execute on root rank (rank 0)
    if cpu_rank != 0:
        return

    # ========== Device Detection ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    else:
        print("\n💻 CPU mode", file=sys.stderr)

    # ========== Check if training should occur ==========
    should_train, start_time_str, current_time_str, elapsed_hours, threshold_hours = (
        check_training_time()
    )
    if not should_train:
        print(
            f"⏸ Waiting for training time (elapsed: {elapsed_hours:.2f}/{threshold_hours:.1f} hours)",
            file=sys.stderr,
        )
        return

    print(
        f"✓ Training enabled (elapsed: {elapsed_hours:.2f}/{threshold_hours:.1f} hours)",
        file=sys.stderr,
    )

    # ========== Lazy model initialization ==========
    if not is_model_initialized:
        net, optimizer, scratch_dir, checkpoint_path = initialize_model(
            device, start_time_str
        )
        is_model_initialized = True
        print("✓ Model initialized", file=sys.stderr)
    else:
        scratch_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"

    # ========== Prepare data for this timestep ==========
    dataloader, pos_np = prepare_data(np.asarray(tas), np.asarray(sfcwind), device)
    if dataloader is None:
        print("⚠ Data preparation failed on non-root rank", file=sys.stderr)
        return

    num_batches = len(dataloader)
    num_samples = (
        len(dataloader.dataset) if hasattr(dataloader.dataset, "__len__") else 0
    )
    sample_size = 2500  # Hardcoded from prepare_data

    # Get global mean tas for monitoring
    if torch.is_tensor(tas):
        global_mean_tas = tas.mean().item()
    else:
        global_mean_tas = np.mean(tas)

    # Get number of nodes from gathered data (approximation on rank 0)
    num_nodes = pos_np.shape[0] if pos_np is not None else 0

    # ========== Run training loop ==========
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Starting training: {num_batches} batch(es)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    losses = run_training_loop(
        net=net,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        num_epochs=1,
    )

    # ========== Report training results ==========
    print(f"\n✓ Training completed", file=sys.stderr)
    print(f"  Final epoch loss: {losses[-1]:.6e}", file=sys.stderr)
    print(f"  Average loss: {np.mean(losses):.6e}", file=sys.stderr)
    print(f"  Total batches: {num_batches}", file=sys.stderr)

    # ========== Save results ==========
    save_training_results(
        net=net,
        optimizer=optimizer,
        losses=losses,
        global_mean_tas=global_mean_tas,
        scratch_dir=scratch_dir,
        start_time_str=start_time_str,
        current_time_str=current_time_str,
        device=device,
        elapsed_hours=elapsed_hours,
        threshold_hours=threshold_hours,
        num_nodes=num_nodes,
        num_samples=num_samples,
        sample_size=sample_size,
    )
