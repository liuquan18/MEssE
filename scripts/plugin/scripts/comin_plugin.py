"""
This is a ComIn Python plugin designed for use in the ICON 2024 training course.
"""

# %%
import comin
import sys

# %%
import numpy as np
import numpy.ma as ma
import json
import glob

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
print("number of domains:", n_dom, file=sys.stderr)


jg = 1  # set the domain id

## primary constructor
# request to register the variable
RHI_MAX_descriptor = ("RHI_MAX", jg)
QI_MAX_descriptor = ("QI_MAX", jg)
log_descriptor = ("log", jg)

comin.var_request_add(RHI_MAX_descriptor, lmodexclusive=False)
comin.var_request_add(QI_MAX_descriptor, lmodexclusive=False)
comin.var_request_add(log_descriptor, lmodexclusive=False)


domain = comin.descrdata_get_domain(jg)
domain_np = np.asarray(domain.cells.decomp_domain)

# no. of local cells (incl. halos):
nc = domain.cells.ncells
print("number of cells:", nc, file=sys.stderr)


# Set metadata
comin.metadata_set(
    RHI_MAX_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Maximum relative humidity over ice",
    units="%",
)

comin.metadata_set(
    QI_MAX_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Maximum cloud ice content",
    units="kg/kg",
)

comin.metadata_set(
    log_descriptor,
    zaxis_id=comin.COMIN_ZAXIS_2D,
    long_name="Log file",
    units="",
)


## secondary constructor
@comin.register_callback(comin.EP_SECONDARY_CONSTRUCTOR)
def simple_python_constructor():
    global RHI_MAX, QI_MAX, temp, qv, exner, qi, log
    RHI_MAX = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        RHI_MAX_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )
    QI_MAX = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        QI_MAX_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )

    log = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE],
        log_descriptor,
        flag=comin.COMIN_FLAG_WRITE,
    )

    temp = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("temp", jg), flag=comin.COMIN_FLAG_READ
    )
    qv = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("qv", jg), flag=comin.COMIN_FLAG_READ
    )
    exner = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("exner", jg), flag=comin.COMIN_FLAG_READ
    )
    qi = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], ("qi", jg), flag=comin.COMIN_FLAG_READ
    )


## help function
# help function to calculate RHI_MAX and QI_MAX
def rhi(temp, qv, p_ex):
    import numpy as np

    rdv = (rd := 287.04) / (rv := 461.51)
    pres = (p0ref := 100000) * np.exp(((cpd := 1004.64) / rd) * np.ma.log(p_ex))
    e_s = 610.78 * np.ma.exp(21.875 * (temp - 273.15) / (temp - 7.66))
    e = pres * qv / (rdv + (1.0 - (rd / rv)) * qv)
    return 100.0 * e / e_s


## callback function
@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_AFTER)
def calculate_rhi_qi():
    # print("simple_python_callbackfct called!", file=sys.stderr)

    # create mask
    mask2d = domain_np != 0
    mask3d = np.repeat(mask2d[:, None, :], domain.nlev, axis=1)

    # apply mask to temp, qv, exner, qi
    temp_np = ma.masked_array(np.squeeze(temp), mask=mask3d)
    qv_np = ma.masked_array(np.squeeze(qv), mask=mask3d)
    exner_np = ma.masked_array(np.squeeze(exner), mask=mask3d)
    qi_np = ma.masked_array(np.squeeze(qi), mask=mask3d)

    # calculate RHI_MAX
    RHI_MAX_np = np.squeeze(RHI_MAX)
    RHI_MAX_3d = rhi(temp_np, qv_np, exner_np)
    RHI_MAX_np[:, :] = np.max(RHI_MAX_3d, axis=1)
    # print("RHI_MAX_np shape", RHI_MAX_np.shape, file=sys.stderr)

    # calculate QI_MAX
    QI_MAX_np = np.squeeze(QI_MAX)
    QI_MAX_np[:, :] = np.max(qi_np, axis=1)
    # print("QI_MAX_np shape", QI_MAX_np.shape, file=sys.stderr)  # (8, 16)


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


# Model and utilities are now imported from separate files


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    global net, optimizer, losses, pos_np, use_gnn  # Declare as global to persist

    RHI_MAX_np_glb = util_gather(np.asarray(RHI_MAX))
    QI_MAX_np_glb = util_gather(np.asarray(QI_MAX))

    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather(cx)
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather(cy)

    print("data gathered!", file=sys.stderr)

    start_time = comin.descrdata_get_simulation_interval().run_start
    start_time_str = (
        str(start_time).replace(" ", "_").replace(":", "-")
    )  # Safe filename

    current_time = comin.current_get_datetime()
    current_time_str = str(current_time).replace(" ", "_").replace(":", "-")

    if rank == 0:

        # Decide whether to use GNN or MLP based on data size
        print("shape of RHI_MAX_np_glb:", RHI_MAX_np_glb.shape, file=sys.stderr)

        num_nodes = len(RHI_MAX_np_glb)

        print("number of nodes:", num_nodes, file=sys.stderr)

        use_gnn = num_nodes > 1000  # Use GNN for large datasets (mini-batch)

        # Check if this is the first timestep
        is_first_timestep = current_time_str == start_time_str
        
        # Calculate elapsed time and check if more than 1 day has passed
        elapsed_time = current_time - start_time
        elapsed_days = elapsed_time.total_seconds() / 86400  # Convert to days
        should_train = elapsed_days > 1.0  # Only train after 1 day
        
        print(f"Elapsed time: {elapsed_time} ({elapsed_days:.2f} days)", file=sys.stderr)
        print(f"Training enabled: {should_train}", file=sys.stderr)

        # Initialize model only at the first timestep
        if is_first_timestep:

            if use_gnn:
                print("=" * 60, file=sys.stderr)
                print("� Initializing Mini-batch GNN model", file=sys.stderr)
                print("=" * 60, file=sys.stderr)

                # Store positions for graph construction
                pos_np = np.column_stack([cx_glb, cy_glb])

                # Initialize GNN model
                net = SimpleGNN(
                    in_channels=1, hidden_channels=32, out_channels=1, num_layers=3
                )
                learning_rate = 0.001
                optimizer = torch.optim.Adam(
                    net.parameters(), lr=learning_rate, weight_decay=1e-5
                )

                print(f"Model: Mini-batch SimpleGNN", file=sys.stderr)
                print(f"  Nodes: {num_nodes}", file=sys.stderr)

            num_params = sum(p.numel() for p in net.parameters())
            print(f"✓ Model initialized at {current_time_str}", file=sys.stderr)
            print(f"  Parameters: {num_params:,}", file=sys.stderr)
            print(f"  Learning rate: {learning_rate}", file=sys.stderr)

        else:
            # Load model and optimizer state at subsequent timesteps
            checkpoint = torch.load(
                f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_str}.pth"
            )

            use_gnn = checkpoint.get("use_gnn", False)

            if use_gnn:
                pos_np = np.column_stack([cx_glb, cy_glb])
                net = SimpleGNN(
                    in_channels=1, hidden_channels=32, out_channels=1, num_layers=3
                )
                optimizer = torch.optim.Adam(
                    net.parameters(), lr=0.001, weight_decay=1e-5
                )

            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(
                f"✓ Model loaded from checkpoint at {current_time_str}", file=sys.stderr
            )

        lossfunc = torch.nn.MSELoss()
        losses = []

        # ============================================
        # Mini-batch GNN Training
        # ============================================

        if use_gnn and should_train:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"🚀 Mini-batch GNN Training on {num_nodes} nodes", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Prepare full data
            x_full = torch.FloatTensor(RHI_MAX_np_glb).unsqueeze(1)  # [num_nodes, 1]
            y_full = torch.FloatTensor(QI_MAX_np_glb).unsqueeze(1)  # [num_nodes, 1]

            # Mini-batch configuration
            batch_size = 5000  # Process 5000 nodes per batch
            num_batches = (num_nodes + batch_size - 1) // batch_size

            print(f"Configuration:", file=sys.stderr)
            print(f"  Batch size: {batch_size} nodes/batch", file=sys.stderr)
            print(f"  Num batches: {num_batches}", file=sys.stderr)
            print(
                f"  Strategy: Spatial blocks with extended neighbors", file=sys.stderr
            )

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
        elif use_gnn and not should_train:
            print(f"\n⏸ Training skipped (waiting for > 1 day elapsed)", file=sys.stderr)
            print(f"  Model initialized but not training yet", file=sys.stderr)

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "use_gnn": use_gnn,
            },
            f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_str}.pth",
        )

        # save log file
        with open(
            f"/scratch/{user[0]}/{user}/icon_exercise_comin/log_{current_time_str}.txt",
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
                    "output_count": len(
                        glob.glob(
                            f"/scratch/{user[0]}/{user}/icon_exercise_comin/log_*.txt"
                        )
                    ),
                },training_enabled": should_train,
                    "elapsed_days": elapsed_days,
                    "
                "training": {
                    "model_type": "GNN (Mini-batch)" if use_gnn else "MLP",
                    "current_loss": float(losses[-1]) if losses else 0.0,
                    "total_batches": len(losses),
                    "batches_per_timestep": len(losses) if losses else 0,
                    "learning_rate": 0.001 if use_gnn else 0.01,
                    "avg_loss": float(np.mean(losses)) if losses else 0.0,
                    "min_loss": float(np.min(losses)) if losses else 0.0,
                    "max_loss": float(np.max(losses)) if losses else 0.0,
                },
            }

            with open(
                f"/scratch/{user[0]}/{user}/icon_exercise_comin/monitor_status.json",
                "w",
            ) as f:
                json.dump(status_data, f, indent=2)

            print(
                f"✓ Monitoring status written to monitor_status.json", file=sys.stderr
            )
        except Exception as e:
            print(f"Warning: Could not write monitoring status: {e}", file=sys.stderr)
