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

import getpass

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


# help function to collect the data from all processes

comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
rank = comm.Get_rank()


def util_gather(data_array: np.ndarray, root=0):

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

    # gather operation
    data_buf = comm.gather((data_array_1d, global_idx), root=root)

    # reorder received data according to global_idx
    if rank == root:
        nglobal = sum([len(gi) for _, gi in data_buf])
        global_array = np.zeros(nglobal, dtype=np.float64)
        for data_array_i, global_idx_i in data_buf:
            global_array[global_idx_i] = data_array_i
        return global_array
    else:
        return None


# training callback
@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    global net, optimizer, losses, pos_np  # Declare as global to persist

    # ============================================
    #      check time condition
    # ===========================================
    # Get timing information first to decide if we should proceed
    start_time_obj = comin.descrdata_get_simulation_interval().run_start
    current_time_obj = comin.current_get_datetime()

    # Convert to strings for filenames
    start_time_str = (
        str(start_time_obj).replace(" ", "_").replace(":", "-")
    )  # Safe filename
    current_time_str = str(current_time_obj).replace(" ", "_").replace(":", "-")

    # Convert to datetime objects for calculations
    start_time = datetime.fromisoformat(str(start_time_obj))
    current_time = datetime.fromisoformat(str(current_time_obj))

    # Calculate elapsed time and check if more than 24 hours have passed
    elapsed_time = current_time - start_time
    elapsed_hours = elapsed_time.total_seconds() / 3600  # Convert to hours
    should_train = elapsed_hours > 1.0  # Only train after 24 hours

    ## Skip training and data gathering
    # Skip everything if training time hasn't arrived yet
    if not should_train:
        if rank == 0:
            print(
                f"\n⏸ Waiting for training time (elapsed: {elapsed_hours:.2f}/2.0 hours)",
                file=sys.stderr,
            )
        return

    # ===========================================
    #           data preparing
    # as an exmple, from temp to sfcwind
    # ===========================================

    # Proceed with data gathering (only when training time has arrived)
    tas_np_glb = util_gather(np.asarray(tas))
    sfcwind_np_glb = util_gather(np.asarray(sfcwind))

    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather(cx)
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather(cy)

    # on process 0, proceed with training
    if rank == 0:

        # data size info
        num_nodes = tas_np_glb.shape[0]
        sample_size = 2500  # Each regional sample contains 2500 nodes

        # Prepare full data (1 batch = 1 timestep from ICON simulation)
        x_full = torch.FloatTensor(tas_np_glb).unsqueeze(1)  # [num_nodes, 1]
        y_full = torch.FloatTensor(sfcwind_np_glb).unsqueeze(1)  # [num_nodes, 1]

        # Store positions for graph construction
        pos_np = np.column_stack([cx_glb, cy_glb])

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
        print(f"\n{'='*60}", file=sys.stderr)
        print(
            f"data gathered!\n input shape {tas_np_glb.shape}, output shape {sfcwind_np_glb.shape}",
            file=sys.stderr,
        )
        print(f"{'='*60}", file=sys.stderr)

        # ===========================================
        #       model initialization / loading
        # ===========================================
        ## prepare directory and data
        # directory to store model checkpoints and logs
        scratch_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
        # Create directory if it doesn't exist (don't delete existing checkpoints)
        os.makedirs(scratch_dir, exist_ok=True)

        # Check if checkpoint exists to decide whether to initialize or load
        checkpoint_path = f"{scratch_dir}/net_{start_time_str}.pth"

        # load models if checkpoint exists
        if os.path.exists(checkpoint_path):
            # Load existing model from checkpoint
            print("=" * 60, file=sys.stderr)
            print("📂 Loading model from checkpoint", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            checkpoint = torch.load(checkpoint_path)

            net = SimpleGNN(
                in_channels=1, hidden_channels=32, out_channels=1, num_layers=3
            )
            learning_rate = 0.001
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate, weight_decay=1e-5
            )

            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(
                f"✓ Model loaded from checkpoint at {current_time_str}", file=sys.stderr
            )
            print(f"  Checkpoint: {checkpoint_path}", file=sys.stderr)

        # initialize new model if no checkpoint
        else:
            # Initialize new model with random weights
            print("=" * 60, file=sys.stderr)
            print("🚀 Initializing new Sample-based GNN model", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            # Initialize GNN model
            net = SimpleGNN(
                in_channels=1, hidden_channels=32, out_channels=1, num_layers=3
            )
            learning_rate = 0.001
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate, weight_decay=1e-5
            )

            num_params = sum(p.numel() for p in net.parameters())
            print(
                f"✓ Model initialized with random weights at {current_time_str}",
                file=sys.stderr,
            )
            print(f"  Parameters: {num_params:,}", file=sys.stderr)
            print(f"  Learning rate: {learning_rate}", file=sys.stderr)

        # ===========================================
        # training
        # ===========================================

        num_batches = len(dataloader)
        num_epochs = 10  # Train for 10 epochs

        lossfunc = torch.nn.MSELoss()
        losses = []

        print(
            f"Training enabled: {should_train}, elapsed hours: {elapsed_hours:.2f}",
            file=sys.stderr,
        )
        # Train over multiple epochs
        for epoch in range(num_epochs):
            epoch_losses = []
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Epoch {epoch+1}/{num_epochs}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Loop over batches (each batch contains multiple samples)
            for batch_idx, (x_list, y_list, edge_list, mask_list) in enumerate(
                dataloader
            ):
                # Merge all samples in the batch into a single big graph
                x_batch, y_batch, edge_batch, mask_batch = batch_graphs(
                    x_list, y_list, edge_list, mask_list
                )

                optimizer.zero_grad()
                y_hat_extended = net(x_batch, edge_batch)
                y_hat = y_hat_extended[mask_batch]

                batch_loss = lossfunc(y_hat, y_batch)
                batch_loss.backward()
                optimizer.step()

                epoch_losses.append(batch_loss.item())
                print(
                    f"  Batch {batch_idx+1}/{num_batches}: Loss = {batch_loss.item():.6e}",
                    file=sys.stderr,
                )

            # Report epoch statistics
            epoch_avg_loss = np.mean(epoch_losses)
            losses.append(epoch_avg_loss)
            print(
                f"  Epoch {epoch+1} completed: Avg Loss = {epoch_avg_loss:.6e}",
                file=sys.stderr,
            )

        print(f"\n✓ GNN training completed", file=sys.stderr)
        print(f"  Final epoch avg loss: {losses[-1]:.6e}", file=sys.stderr)
        print(f"  Overall avg loss: {np.mean(losses):.6e}", file=sys.stderr)
        print(
            f"  Total epochs: {num_epochs}, Batches per epoch: {num_batches}",
            file=sys.stderr,
        )

        # ===========================================
        #       save model and logs
        # ===========================================
        if "net" in locals():
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{scratch_dir}/net_{start_time_str}.pth",
            )

        # save log file with detailed batch losses and epoch averages
        avg_loss = np.mean(losses)
        with open(
            f"{scratch_dir}/log_{current_time_str}.txt",
            "w",
        ) as f:
            # Write epoch average losses (one per epoch)
            for epoch_loss in losses:
                f.write(f"{epoch_loss}\n")

        # Also save detailed batch-level losses for analysis
        with open(
            f"{scratch_dir}/log_detailed_{current_time_str}.txt",
            "w",
        ) as f:
            f.write(f"# Epoch-level average losses\n")
            for i, epoch_loss in enumerate(losses):
                f.write(f"{epoch_loss:.6e}\n")

        # Write monitoring status JSON for real-time monitoring
        try:
            import json

            elapsed_time = current_time - start_time

            status_data = {
                "timestamp": current_time.isoformat(),
                "simulation": {
                    "start_time": str(start_time),
                    "current_time": str(current_time),
                    "elapsed_time": str(elapsed_time),
                    "n_domains": int(n_dom) if n_dom is not None else 1,
                    "total_points": num_nodes,
                    "output_count": len(glob.glob(f"{scratch_dir}/log_*.txt")),
                },
                "training": {
                    "model_type": "GNN (Sample-based)",
                    "training_enabled": should_train,
                    "elapsed_hours": elapsed_hours,
                    "current_loss": float(losses[-1]) if losses else 0.0,
                    "num_epochs": num_epochs if "num_epochs" in locals() else 10,
                    "num_batches": num_batches if "num_batches" in locals() else 0,
                    "batches_per_epoch": (
                        num_batches if "num_batches" in locals() else 0
                    ),
                    "total_batches": (
                        num_epochs * num_batches
                        if "num_epochs" in locals() and "num_batches" in locals()
                        else 0
                    ),
                    "batches_per_timestep": (
                        num_epochs * num_batches
                        if "num_epochs" in locals() and "num_batches" in locals()
                        else 0
                    ),
                    "num_samples": num_samples if "num_samples" in locals() else 0,
                    "sample_size": sample_size if "sample_size" in locals() else 2500,
                    "learning_rate": 0.001,
                    "avg_loss": float(np.mean(losses)) if losses else 0.0,
                    "min_loss": float(np.min(losses)) if losses else 0.0,
                    "max_loss": float(np.max(losses)) if losses else 0.0,
                },
            }

            with open(
                f"{scratch_dir}/monitor_status.json",
                "w",
            ) as f:
                json.dump(status_data, f, indent=2)

            print(
                f"✓ Monitoring status written to monitor_status.json", file=sys.stderr
            )
        except Exception as e:
            print(f"Warning: Could not write monitoring status: {e}", file=sys.stderr)
