# GNN Model for Fraud Detection
# Implements a simple Graph Neural Network for transaction graph data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

# Justification: GNNs capture relational and structural information in transaction graphs, but may miss temporal/sequential patterns critical for fraud detection.
