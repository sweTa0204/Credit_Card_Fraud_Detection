# LSTM Model for Fraud Detection
# Implements a simple LSTM for sequential transaction data

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, seq_feat_dim, hidden_dim, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(seq_feat_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        out = self.fc(pooled)
        return out

# Justification: LSTMs capture temporal dependencies in transaction sequences, but ignore graph-based relationships between users, merchants, and transactions.
