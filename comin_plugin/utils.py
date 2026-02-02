"""
Utility functions for graph construction and batch processing on ICON grid.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RegionalSampleDataset(Dataset):
    """Dataset for regional samples from ICON grid."""

    def __init__(self, x_full, y_full, pos_np, sample_size=2500, k=6, extended_k=8):
        """
        Parameters:
        -----------
        x_full : torch.Tensor
            Full input data [num_nodes, features]
        y_full : torch.Tensor
            Full target data [num_nodes, features]
        pos_np : np.ndarray
            Node positions [num_nodes, 2]
        sample_size : int
            Number of nodes per regional sample
        k : int
            Number of nearest neighbors
        extended_k : int
            Extended neighbors for context
        """
        self.x_full = x_full
        self.y_full = y_full
        self.pos_np = pos_np
        self.sample_size = sample_size
        self.k = k
        self.extended_k = extended_k

        self.num_nodes = x_full.shape[0]
        self.num_samples = (self.num_nodes + sample_size - 1) // sample_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_node = idx * self.sample_size
        end_node = min(start_node + self.sample_size, self.num_nodes)
        sample_node_range = range(start_node, end_node)

        # Build subgraph for this sample
        sample_edge_index_np, sample_node_ids = build_knn_graph_batch_numpy(
            self.pos_np, sample_node_range, k=self.k, extended_k=self.extended_k
        )

        # Convert to tensors
        sample_edge_index = torch.LongTensor(sample_edge_index_np)
        x_sample = self.x_full[sample_node_ids]
        y_sample = self.y_full[sample_node_range]

        # Create target mask
        target_mask = torch.isin(
            torch.LongTensor(sample_node_ids),
            torch.LongTensor(list(sample_node_range)),
        )

        return x_sample, y_sample, sample_edge_index, target_mask


def collate_samples(batch):
    """Collate function to keep variable-sized graph samples in lists."""
    x_list, y_list, edge_list, mask_list = zip(*batch)
    return list(x_list), list(y_list), list(edge_list), list(mask_list)


def batch_graphs(x_list, y_list, edge_list, mask_list):
    """Combine variable-sized graph samples into one big graph for a single forward/backward.

    Offsets edge indices per sample and concatenates node features, targets, and masks.
    """
    offsets = []
    total = 0
    for x in x_list:
        offsets.append(total)
        total += x.shape[0]

    x_cat = torch.cat(x_list, dim=0)
    y_cat = torch.cat(y_list, dim=0)
    mask_cat = torch.cat(mask_list, dim=0)

    edge_cat_list = []
    for offset, edge in zip(offsets, edge_list):
        edge_cat_list.append(edge + offset)
    edge_cat = (
        torch.cat(edge_cat_list, dim=1)
        if edge_cat_list
        else torch.zeros((2, 0), dtype=torch.long)
    )

    return x_cat, y_cat, edge_cat, mask_cat


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


def process_batches_for_inference(
    data, pos, model, batch_size=5000, k=6, extended_k=8, device="cpu"
):
    """
    Process data in mini-batches for GNN inference.

    Parameters:
    -----------
    data : np.ndarray or torch.Tensor
        Input data [num_nodes, features]
    pos : np.ndarray
        Node positions [num_nodes, 2] (lon, lat)
    model : torch.nn.Module
        Trained GNN model
    batch_size : int
        Number of nodes per batch
    k : int
        Number of nearest neighbors for graph construction
    extended_k : int
        Extended neighbors for context
    device : str
        Device for inference ('cpu' or 'cuda')

    Returns:
    --------
    predictions : np.ndarray
        Predictions for all nodes [num_nodes, out_features]
    """
    # Convert data to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)

    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    num_nodes = data.shape[0]
    num_batches = (num_nodes + batch_size - 1) // batch_size

    # Initialize output array
    predictions = torch.zeros(num_nodes, model.node_mlps[-1].out_features)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_node = batch_idx * batch_size
            end_node = min(start_node + batch_size, num_nodes)
            batch_node_range = range(start_node, end_node)

            # Build subgraph for this batch
            batch_edge_index_np, batch_node_ids = build_knn_graph_batch_numpy(
                pos, batch_node_range, k=k, extended_k=extended_k
            )

            # Convert to tensors
            batch_edge_index = torch.LongTensor(batch_edge_index_np).to(device)
            x_batch = data[batch_node_ids].to(device)

            # Forward pass
            y_hat_extended = model(x_batch, batch_edge_index)

            # Extract predictions for target nodes
            target_mask = torch.isin(
                torch.LongTensor(batch_node_ids),
                torch.LongTensor(list(batch_node_range)),
            )

            # Map back to original positions
            target_indices = np.array(batch_node_ids)[target_mask.cpu().numpy()]
            predictions[target_indices] = y_hat_extended[target_mask].cpu()

    return predictions.numpy()


def load_model_checkpoint(checkpoint_path, model_class, model_kwargs=None):
    """
    Load a trained model from checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    model_class : class
        Model class (e.g., SimpleGNN)
    model_kwargs : dict
        Keyword arguments for model initialization

    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    checkpoint : dict
        Full checkpoint dictionary
    """
    if model_kwargs is None:
        model_kwargs = {
            "in_channels": 1,
            "hidden_channels": 32,
            "out_channels": 1,
            "num_layers": 3,
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint
