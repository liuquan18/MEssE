""" 
GNN-enhanced ComIn Python plugin for ICON model
Using Graph Neural Networks to leverage spatial structure of ICON grid
"""

# %%
import comin
import sys
#%%
import numpy as np
import numpy.ma as ma
import pandas as pd

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

# PyTorch Geometric imports for GNN
try:
    from torch_geometric.nn import GCNConv, GATConv, MessagePassing
    from torch_geometric.data import Data
    from torch_geometric.utils import knn_graph
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric is available - GNN mode enabled", file=sys.stderr)
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not found - falling back to MLP mode", file=sys.stderr)

import getpass
user = getpass.getuser()
torch.manual_seed(0)

# Debugging: Print number of domains
glob = comin.descrdata_get_global()
n_dom = glob.n_dom
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


def build_knn_graph(pos, k=6):
    """
    Build k-nearest neighbor graph from spatial coordinates.
    ICON icosahedral grid typically has 6 neighbors for most cells.
    
    Parameters:
    -----------
    pos : torch.Tensor
        Node positions [num_nodes, 2] (lon, lat)
    k : int
        Number of nearest neighbors (default: 6 for ICON grid)
    
    Returns:
    --------
    edge_index : torch.Tensor
        Graph connectivity [2, num_edges]
    """
    edge_index = knn_graph(pos, k=k, loop=False)
    return edge_index


# GNN Model Definition
class GNNNet(nn.Module):
    """
    Graph Neural Network for ICON grid data.
    Uses Graph Attention Network (GAT) layers to model spatial dependencies.
    """
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, num_layers=3, heads=4):
        super(GNNNet, self).__init__()
        
        self.num_layers = num_layers
        
        # First layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.1)
        
        # Middle layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1)
            )
        
        # Output layer (single head for final prediction)
        self.conv_out = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        """
        Forward pass through GNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, in_channels]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        
        Returns:
        --------
        x : torch.Tensor
            Node predictions [num_nodes, out_channels]
        """
        # First layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Middle layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.conv_out(x, edge_index)
        
        return x


# Fallback MLP model (if PyTorch Geometric not available)
class MLPNet(nn.Module):
    """Fallback Multi-Layer Perceptron"""
    def __init__(self, n_inputs=30, n_outputs=30, n_hidden=32):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@comin.register_callback(comin.EP_ATM_WRITE_OUTPUT_BEFORE)
def get_batch_callback():
    global net, optimizer, losses, edge_index, use_gnn
    
    RHI_MAX_np_glb = util_gather(np.asarray(RHI_MAX))
    QI_MAX_np_glb = util_gather(np.asarray(QI_MAX))
    
    # Get cell coordinates (longitude, latitude)
    cx = np.rad2deg(domain.cells.clon)
    cx_glb = util_gather(cx)
    cy = np.rad2deg(domain.cells.clat)
    cy_glb = util_gather(cy)
    
    print("data gathered!", file=sys.stderr)

    start_time = comin.descrdata_get_simulation_interval().run_start
    start_time_np = pd.to_datetime(start_time)

    current_time = comin.current_get_datetime()
    current_time_np = pd.to_datetime(current_time)

    if rank == 0:
        
        # Determine if we use GNN or MLP
        use_gnn = TORCH_GEOMETRIC_AVAILABLE
        
        # Initialize model only at the first timestep
        if current_time_np == start_time_np:
            
            if use_gnn:
                print("=" * 60, file=sys.stderr)
                print("Initializing GNN model (Graph Neural Network)", file=sys.stderr)
                print("=" * 60, file=sys.stderr)
                
                # Build spatial graph structure from coordinates
                pos = torch.FloatTensor(np.column_stack([cx_glb, cy_glb]))
                num_nodes = pos.shape[0]
                
                # Build k-NN graph (k=6 for ICON icosahedral grid)
                edge_index = build_knn_graph(pos, k=6)
                print(f"Graph built: {num_nodes} nodes, {edge_index.shape[1]} edges", file=sys.stderr)
                
                # Initialize GNN model
                net = GNNNet(
                    in_channels=1,      # Input: RHI_MAX per node
                    hidden_channels=32, # Hidden dimension
                    out_channels=1,     # Output: QI_MAX per node
                    num_layers=3,       # Number of GNN layers
                    heads=4             # Number of attention heads
                )
                learning_rate = 0.001  # Lower LR for GNN (more stable)
                
            else:
                print("Initializing MLP model (fallback mode)", file=sys.stderr)
                net = MLPNet(n_inputs=30, n_outputs=30, n_hidden=32)
                learning_rate = 0.01
            
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
            print(f"Model initialized at {current_time_np}", file=sys.stderr)
            print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}", file=sys.stderr)
            
        else:
            # Load model and optimizer state at subsequent timesteps
            checkpoint = torch.load(
                f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_np}.pth"
            )
            
            use_gnn = checkpoint.get('use_gnn', False)
            
            if use_gnn:
                if not TORCH_GEOMETRIC_AVAILABLE:
                    print("WARNING: Checkpoint uses GNN but PyTorch Geometric not available!", file=sys.stderr)
                    use_gnn = False
                    net = MLPNet()
                    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
                else:
                    # Rebuild graph structure
                    pos = torch.FloatTensor(np.column_stack([cx_glb, cy_glb]))
                    edge_index = build_knn_graph(pos, k=6)
                    
                    net = GNNNet(in_channels=1, hidden_channels=32, out_channels=1, num_layers=3, heads=4)
                    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
            else:
                net = MLPNet()
                optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Model loaded from checkpoint at {current_time_np}", file=sys.stderr)

        lossfunc = torch.nn.MSELoss()
        losses = []
        
        # Training loop
        if use_gnn:
            # ============================================
            # GNN Training: Treat entire graph as one batch
            # ============================================
            print(f"GNN Training on {len(RHI_MAX_np_glb)} nodes", file=sys.stderr)
            
            # Prepare graph data
            x = torch.FloatTensor(RHI_MAX_np_glb).unsqueeze(1)  # [num_nodes, 1]
            y = torch.FloatTensor(QI_MAX_np_glb).unsqueeze(1)   # [num_nodes, 1]
            
            # Multiple training iterations per timestep
            num_epochs_per_timestep = 10
            
            for epoch in range(num_epochs_per_timestep):
                optimizer.zero_grad()
                
                # Forward pass through GNN
                y_hat = net(x, edge_index)
                
                # Compute loss
                loss = lossfunc(y_hat, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % 3 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs_per_timestep}, Loss: {loss.item():.6f}", file=sys.stderr)
            
            print(f"GNN training completed. Final loss: {losses[-1]:.6f}", file=sys.stderr)
            
        else:
            # ============================================
            # MLP Training: Batch processing (original approach)
            # ============================================
            B = 5  # batch size
            C = 1  # channel
            H = 30  # height
            
            one_batch_size = B * C * H
            total_size = RHI_MAX_np_glb.shape[0]
            num_batches = total_size // one_batch_size
            
            print(f"MLP Training: {num_batches} batches", file=sys.stderr)
            
            for i in range(num_batches):
                x_batch_np = RHI_MAX_np_glb[i * one_batch_size : (i + 1) * one_batch_size]
                y_batch_np = QI_MAX_np_glb[i * one_batch_size : (i + 1) * one_batch_size]
                
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
                print(f"  Batch {i+1}/{num_batches}, Loss: {loss.item():.6f}", file=sys.stderr)

        # Save checkpoint
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'use_gnn': use_gnn,
        }, f"/scratch/{user[0]}/{user}/icon_exercise_comin/net_{start_time_np}.pth")

        # Save log file
        with open(
            f"/scratch/{user[0]}/{user}/icon_exercise_comin/log_{current_time_np}.txt",
            "w",
        ) as f:
            for item in losses:
                f.write("%s\n" % item)
        
        print(f"Checkpoint saved. Average loss: {np.mean(losses):.6f}", file=sys.stderr)
