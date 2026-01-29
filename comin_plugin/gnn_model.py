"""
Graph Neural Network model definition for ICON grid data processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
