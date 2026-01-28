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


def build_knn_graph_batch_numpy(pos, batch_nodes, k=6, extended_k=8):
    """
    Build k-NN graph for a batch of nodes with their neighbors (mini-batch strategy).

    Strategy:
    1. Select batch nodes
    2. Find extended neighbors to avoid boundary issues
    3. Build edges within extended node set

    Parameters:
    -----------
    pos : np.ndarray
        All node positions [num_nodes, 2]
    batch_nodes : np.ndarray or range
        Indices of nodes in current batch
    k : int
        Number of nearest neighbors
    extended_k : int
        Extended neighbors for context (default: k+2)

    Returns:
    --------
    batch_edge_index : np.ndarray
        Edge indices relative to batch [2, num_edges]
    batch_node_ids : np.ndarray
        Global node IDs in batch (batch + neighbors)
    """
    batch_nodes = np.array(batch_nodes)
    num_all_nodes = pos.shape[0]

    # Find extended neighbors for batch nodes
    extended_nodes_set = set(batch_nodes)

    for node_id in batch_nodes:
        # Compute distances from this node to all others
        dists = np.sum((pos - pos[node_id : node_id + 1]) ** 2, axis=1)
        # Find k+extended_k nearest neighbors
        nearest = np.argpartition(dists, min(extended_k + 1, num_all_nodes))[
            : extended_k + 1
        ]
        extended_nodes_set.update(nearest[nearest != node_id])

    # Convert to sorted array
    batch_node_ids = np.array(sorted(extended_nodes_set))
    num_batch_nodes = len(batch_node_ids)

    # Build mapping from global ID to local batch ID
    global_to_local = {gid: lid for lid, gid in enumerate(batch_node_ids)}

    # Build k-NN edges within extended batch
    pos_batch = pos[batch_node_ids]
    edges_list = []

    for local_i, global_i in enumerate(batch_node_ids):
        # Compute distances within batch
        dists = np.sum((pos_batch - pos_batch[local_i : local_i + 1]) ** 2, axis=1)
        # Find k nearest neighbors
        nearest_local = np.argpartition(dists, min(k + 1, num_batch_nodes))[
            : min(k + 1, num_batch_nodes)
        ]
        nearest_local = nearest_local[nearest_local != local_i][:k]

        # Add edges
        for local_j in nearest_local:
            edges_list.append([local_i, local_j])

    batch_edge_index = (
        np.array(edges_list).T if edges_list else np.zeros((2, 0), dtype=np.int64)
    )

    return batch_edge_index, batch_node_ids


def interpolate_to_regular_grid(
    cx_glb, cy_glb, data_glb, resolution=0.1, method="linear"
):
    """
    Interpolate unstructured ICON grid data to a regular lat-lon grid.

    Parameters:
    -----------
    cx_glb : np.ndarray
        1D array of longitudes (degrees)
    cy_glb : np.ndarray
        1D array of latitudes (degrees)
    data_glb : np.ndarray
        1D array of data values
    resolution : float
        Grid resolution in degrees (default: 0.1)
    method : str
        Interpolation method: 'linear', 'nearest', or 'cubic' (default: 'linear')

    Returns:
    --------
    lon_grid : np.ndarray
        2D array of longitudes
    lat_grid : np.ndarray
        2D array of latitudes
    data_grid : np.ndarray
        2D array of interpolated data
    """
    from scipy.interpolate import griddata

    # Define regular grid
    lon_min, lon_max = cx_glb.min(), cx_glb.max()
    lat_min, lat_max = cy_glb.min(), cy_glb.max()

    lon_1d = np.arange(lon_min, lon_max, resolution)
    lat_1d = np.arange(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # Interpolate
    points = np.column_stack((cx_glb, cy_glb))
    data_grid = griddata(points, data_glb, (lon_grid, lat_grid), method=method)

    return lon_grid, lat_grid, data_grid


# Simple Graph Neural Network using only PyTorch (no PyG dependency)
class SimpleGNN(nn.Module):
    """
    Lightweight Graph Neural Network using pure PyTorch.
    Implements message passing manually without PyTorch Geometric.
    """

    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, num_layers=3):
        super(SimpleGNN, self).__init__()

        self.num_layers = num_layers

        # Node transformation layers
        self.node_mlps = nn.ModuleList()
        self.node_mlps.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.node_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        self.node_mlps.append(nn.Linear(hidden_channels, out_channels))

        # Message passing layers (aggregate neighbor information)
        self.message_mlps = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.message_mlps.append(nn.Linear(hidden_channels * 2, hidden_channels))

        self.dropout = nn.Dropout(0.1)

    def message_passing(self, x, edge_index, message_mlp):
        """
        Perform one round of message passing.

        Parameters:
        -----------
        x : torch.Tensor [num_nodes, features]
        edge_index : torch.Tensor [2, num_edges]
        message_mlp : nn.Module

        Returns:
        --------
        aggregated : torch.Tensor [num_nodes, features]
        """
        src, dst = edge_index[0], edge_index[1]

        # Gather neighbor features
        src_features = x[src]  # [num_edges, features]
        dst_features = x[dst]  # [num_edges, features]

        # Concatenate source and destination features
        messages = torch.cat(
            [src_features, dst_features], dim=1
        )  # [num_edges, 2*features]

        # Transform messages
        messages = message_mlp(messages)  # [num_edges, features]

        # Aggregate messages for each node (sum aggregation)
        num_nodes = x.shape[0]
        aggregated = torch.zeros(num_nodes, messages.shape[1], device=x.device)
        aggregated.index_add_(0, dst, messages)

        # Normalize by number of neighbors (optional)
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = degree.clamp(min=1).unsqueeze(1)
        aggregated = aggregated / degree

        return aggregated

    def forward(self, x, edge_index):
        """
        Forward pass through GNN.

        Parameters:
        -----------
        x : torch.Tensor [num_nodes, in_channels]
        edge_index : torch.Tensor [2, num_edges]

        Returns:
        --------
        x : torch.Tensor [num_nodes, out_channels]
        """
        # First layer: node transformation
        x = self.node_mlps[0](x)
        x = F.relu(x)
        x = self.dropout(x)

        # Middle layers: message passing + node update
        for i in range(self.num_layers - 2):
            # Message passing
            messages = self.message_passing(x, edge_index, self.message_mlps[i])

            # Update node features (residual connection)
            x = x + messages
            x = self.node_mlps[i + 1](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Output layer
        x = self.node_mlps[-1](x)

        return x


class Net(nn.Module):
    def __init__(self, n_inputs=30, n_outputs=30, n_hidden=32):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_inputs, n_hidden)  # 5*5 from image dimension
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

            else:
                print("Initializing MLP model", file=sys.stderr)
                net = Net(n_inputs=30, n_outputs=30, n_hidden=32)
                learning_rate = 0.01
                optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

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
            else:
                net = Net()
                optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

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

        if use_gnn:
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

        else:
            # ============================================
            # MLP Training: Batch processing
            # ============================================
            B = 5  # batch size
            C = 1  # channel
            H = 30  # height

            one_batch_size = B * C * H
            total_size = RHI_MAX_np_glb.shape[0]
            num_batches = total_size // one_batch_size

            print(f"MLP Training: {num_batches} batches", file=sys.stderr)

            for i in range(num_batches):
                x_batch_np = RHI_MAX_np_glb[
                    i * one_batch_size : (i + 1) * one_batch_size
                ]
                y_batch_np = QI_MAX_np_glb[
                    i * one_batch_size : (i + 1) * one_batch_size
                ]

                # reshape
                x_batch_np = x_batch_np.reshape(B, C, H)
                y_batch_np = y_batch_np.reshape(B, C, H)

                # to tensor
                x_batch = torch.FloatTensor(x_batch_np)
                y_batch = torch.FloatTensor(y_batch_np)

                # train
                optimizer.zero_grad()
                y_hat = net(x_batch)
                loss = lossfunc(y_hat, y_batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                if i % 50 == 0:
                    print(
                        f"  Batch {i+1}/{num_batches}: Loss = {loss.item():.6e}",
                        file=sys.stderr,
                    )

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
                },
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
