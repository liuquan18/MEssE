"""
This is a ComIn Python plugin designed for use in the ICON 2024 training course.
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
from datetime import datetime

# %%
import torch

import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

import getpass

# Import model and utilities
from gnn_model import SimpleGNN
from utils import build_knn_graph_batch_numpy

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
    global temp, sfcwind, domain
    # Read existing ICON variables directly (no need for descriptors since they're not registered)
    temp = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", jg), flag=comin.COMIN_FLAG_READ
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

    ## about time
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
    should_train = elapsed_hours > 24.0  # Only train after 24 hours

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
    #           data gathering
    # as an exmple, from temp to sfcwind
    # ===========================================

    # Proceed with data gathering (only when training time has arrived)
    temp_np_glb = util_gather(np.asarray(temp))
    sfcwind_np_glb = util_gather(np.asarray(sfcwind))

    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather(cx)
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather(cy)

    # on process 0, proceed with training
    if rank == 0:

        print(
            f"data gathered! input shape {temp_np_glb.shape}, output shape {sfcwind_np_glb.shape}",
            file=sys.stderr,
        )
        print(
            f"Training enabled: {should_train}, elapsed hours: {elapsed_hours:.2f}",
            file=sys.stderr,
        )

        ## prepare directory and data
        # directory to store model checkpoints and logs
        scratch_dir = f"/scratch/{user[0]}/{user}/icon_exercise_comin"
        # Remove directory and all contents if it exists, then recreate it
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)
        os.makedirs(scratch_dir)

        # Store positions for graph construction
        pos_np = np.column_stack([cx_glb, cy_glb])

        # Check if checkpoint exists to decide whether to initialize or load
        checkpoint_path = f"{scratch_dir}/net_{start_time_str}.pth"

        # ===========================================
        #       model initialization / loading
        # ===========================================

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
            print("🚀 Initializing new Mini-batch GNN model", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            # Initialize GNN model
            net = SimpleGNN(
                in_channels=1, hidden_channels=32, out_channels=1, num_layers=3
            )
            learning_rate = 0.001
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate, weight_decay=1e-5
            )

            print(f"Model: Mini-batch SimpleGNN", file=sys.stderr)

            num_params = sum(p.numel() for p in net.parameters())
            print(
                f"✓ Model initialized with random weights at {current_time_str}",
                file=sys.stderr,
            )
            print(f"  Parameters: {num_params:,}", file=sys.stderr)
            print(f"  Learning rate: {learning_rate}", file=sys.stderr)

        # ===========================================
        # mini-batch training
        # ===========================================

        # Mini-batch configuration
        num_nodes = temp_np_glb.shape[0]
        batch_size = 2000  # Process 2000 nodes per batch
        num_batches = (num_nodes + batch_size - 1) // batch_size

        lossfunc = torch.nn.MSELoss()
        losses = []

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"🚀 Mini-batch GNN Training on {num_nodes} nodes", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Prepare full data
        x_full = torch.FloatTensor(temp_np_glb).unsqueeze(1)  # [num_nodes, 1]
        y_full = torch.FloatTensor(sfcwind_np_glb).unsqueeze(1)  # [num_nodes, 1]

        print(f"Configuration:", file=sys.stderr)
        print(f"  Batch size: {batch_size} nodes/batch", file=sys.stderr)
        print(f"  Num batches: {num_batches}", file=sys.stderr)
        print(f"  Strategy: Spatial blocks with extended neighbors", file=sys.stderr)

        # Train on each batch
        for batch_idx in range(num_batches):
            start_node = batch_idx * batch_size
            end_node = min(start_node + batch_size, num_nodes)
            batch_node_range = range(start_node, end_node)

            print(
                f"\n  📦 Batch {batch_idx+1}/{num_batches}: nodes [{start_node}:{end_node}]",
                file=sys.stderr,
            )

            # Build subgraph for this batch
            batch_edge_index_np, batch_node_ids = build_knn_graph_batch_numpy(
                pos_np, batch_node_range, k=6, extended_k=8
            )

            # Convert to tensors
            batch_edge_index = torch.LongTensor(batch_edge_index_np)
            x_batch = x_full[batch_node_ids]
            y_batch = y_full[batch_node_range]  # Only compute loss on target nodes

            num_batch_edges = batch_edge_index.shape[1]
            print(
                f"     Subgraph: {len(batch_node_ids)} nodes, {num_batch_edges} edges",
                file=sys.stderr,
            )

            # Forward pass
            optimizer.zero_grad()
            y_hat_extended = net(x_batch, batch_edge_index)

            # Extract predictions for target nodes (first len(batch_node_range) nodes)
            target_mask = torch.isin(
                torch.LongTensor(batch_node_ids),
                torch.LongTensor(list(batch_node_range)),
            )
            y_hat = y_hat_extended[target_mask]

            # Compute loss and update
            loss = lossfunc(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(f"     Loss: {loss.item():.6e}", file=sys.stderr)

        print(f"\n✓ Mini-batch GNN training completed", file=sys.stderr)
        print(f"  Final loss: {losses[-1]:.6e}", file=sys.stderr)
        print(f"  Average loss: {np.mean(losses):.6e}", file=sys.stderr)
        print(f"  Total batches processed: {num_batches}", file=sys.stderr)

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

        # save log file
        with open(
            f"{scratch_dir}/log_{current_time_str}.txt",
            "w",
        ) as f:
            for item in losses:
                f.write("%s\n" % item)

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
                    "n_domains": (
                        int(n_dom[0]) if hasattr(n_dom, "__len__") else int(n_dom)
                    ),
                    "total_points": num_nodes,
                    "output_count": len(glob.glob(f"{scratch_dir}/log_*.txt")),
                },
                "training": {
                    "model_type": "GNN (Mini-batch)",
                    "training_enabled": should_train,
                    "elapsed_hours": elapsed_hours,
                    "current_loss": float(losses[-1]) if losses else 0.0,
                    "total_batches": len(losses),
                    "batches_per_timestep": len(losses) if losses else 0,
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
