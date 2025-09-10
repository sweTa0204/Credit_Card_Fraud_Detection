# Model Evaluation Script: GNN, LSTM, and Hybrid Comparison
# This script trains and evaluates each model, saves results, and plots comparison

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from gnn_model import SimpleGNN
from lstm_model import SimpleLSTM
from hybrid_model_attentiongnn_lstm import AttentionGNNLSTM

# Load data
features = pd.read_csv('engineered_features.csv')
labels = features['Fraud_Flag'].values


# For demonstration, use only numeric features (excluding IDs, dates, etc.)
X = features.select_dtypes(include=[np.number]).drop('Fraud_Flag', axis=1)
# Impute missing values with mean
X = X.fillna(X.mean())
X = X.values


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)

# Apply SMOTE oversampling to the training set
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Dummy tensor conversion for demonstration
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Model configs
input_dim = X_train.shape[1]
hidden_dim = 32
num_classes = 2


# --- Class Imbalance Handling ---
from collections import Counter
class_counts = Counter(y_train)
class_weights = torch.tensor([
    class_counts[0] / len(y_train),
    class_counts[1] / len(y_train)
], dtype=torch.float32)
class_weights = 1.0 / class_weights
class_weights = class_weights / class_weights.sum()

# --- LSTM Model ---
lstm_model = SimpleLSTM(seq_feat_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# For demonstration, treat each sample as a sequence of length 1 with feature_dim = input_dim
X_train_seq = X_train_tensor.unsqueeze(1)  # shape: (batch, 1, input_dim)
X_test_seq = X_test_tensor.unsqueeze(1)


# Train LSTM
for epoch in range(10):
    lstm_model.train()
    lstm_optimizer.zero_grad()
    output = lstm_model(X_train_seq)
    print(f"LSTM output shape: {output.shape}, y_train_tensor shape: {y_train_tensor.shape}")
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    lstm_optimizer.step()


lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = lstm_model(X_test_seq).argmax(dim=1).numpy()
    y_score_lstm = torch.softmax(lstm_model(X_test_seq), dim=1)[:,1].numpy()



# --- GNN Model (process each sample as its own graph) ---
gnn_model = SimpleGNN(node_feat_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

def gnn_forward_batch(model, X_tensor):
    outputs = []
    for i in range(X_tensor.shape[0]):
        x = X_tensor[i].unsqueeze(0)  # (1, input_dim)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)  # dummy self-loop
        batch = torch.zeros(1, dtype=torch.long)
        out = model(x, edge_index, batch)
        outputs.append(out)
    return torch.cat(outputs, dim=0)

for epoch in range(10):
    gnn_model.train()
    gnn_optimizer.zero_grad()
    output = gnn_forward_batch(gnn_model, X_train_tensor)
    print(f"GNN output shape: {output.shape}, y_train_tensor shape: {y_train_tensor.shape}")
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    gnn_optimizer.step()

gnn_model.eval()
with torch.no_grad():
    gnn_out = gnn_forward_batch(gnn_model, X_test_tensor)
    y_pred_gnn = gnn_out.argmax(dim=1).numpy()
    y_score_gnn = torch.softmax(gnn_out, dim=1)[:,1].numpy()



# --- Hybrid Model (process each sample as its own graph/sequence) ---
hybrid_model = AttentionGNNLSTM(node_feat_dim=input_dim, seq_feat_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
hybrid_optimizer = optim.Adam(hybrid_model.parameters(), lr=0.001)

def hybrid_forward_batch(model, X_tensor, X_seq_tensor):
    outputs = []
    for i in range(X_tensor.shape[0]):
        x = X_tensor[i].unsqueeze(0)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long)
        seq_x = X_seq_tensor[i].unsqueeze(0)
        out = model(x, edge_index, batch, seq_x)
        outputs.append(out)
    return torch.cat(outputs, dim=0)

for epoch in range(10):
    hybrid_model.train()
    hybrid_optimizer.zero_grad()
    output = hybrid_forward_batch(hybrid_model, X_train_tensor, X_train_seq)
    print(f"Hybrid output shape: {output.shape}, y_train_tensor shape: {y_train_tensor.shape}")
    loss = loss_fn(output, y_train_tensor)
    loss.backward()
    hybrid_optimizer.step()



# --- Evaluation ---
def get_metrics(y_true, y_pred, y_score):
    print('y_pred:', y_pred)
    print('y_score:', y_score)
    if np.isnan(y_pred).any() or np.isnan(y_score).any():
        print('NaN detected in predictions or scores!')
        return {'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'AUC': np.nan}
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score)
    }

# --- Hybrid Model Evaluation at Multiple Thresholds ---
hybrid_model.eval()
with torch.no_grad():
    hybrid_out = hybrid_forward_batch(hybrid_model, X_test_tensor, X_test_seq)
    y_score_hybrid = torch.softmax(hybrid_out, dim=1)[:,1].numpy()
    # Default threshold (0.5)
    y_pred_hybrid = (y_score_hybrid > 0.5).astype(int)

# --- Evaluate Hybrid Model at Multiple Thresholds ---
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
hybrid_threshold_results = {}
for thresh in thresholds:
    y_pred_thresh = (y_score_hybrid > thresh).astype(int)
    hybrid_threshold_results[thresh] = get_metrics(y_test, y_pred_thresh, y_score_hybrid)

print("\nHybrid Model Precision/Recall at Different Thresholds:")
for thresh in thresholds:
    res = hybrid_threshold_results[thresh]
    print(f"Threshold {thresh:.2f}: Precision={res['Precision']:.3f}, Recall={res['Recall']:.3f}, F1={res['F1']:.3f}, AUC={res['AUC']:.3f}")

# --- Evaluation ---
def get_metrics(y_true, y_pred, y_score):
    print('y_pred:', y_pred)
    print('y_score:', y_score)
    if np.isnan(y_pred).any() or np.isnan(y_score).any():
        print('NaN detected in predictions or scores!')
        return {'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'AUC': np.nan}
    return {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score)
    }


results = {
    'LSTM': get_metrics(y_test, y_pred_lstm, y_score_lstm),
    'GNN': get_metrics(y_test, y_pred_gnn, y_score_gnn),
    'Hybrid': get_metrics(y_test, y_pred_hybrid, y_score_hybrid)
}

# Save hybrid threshold results for documentation
pd.DataFrame(hybrid_threshold_results).T.to_csv('hybrid_threshold_results.csv')

# --- Plot Results ---
labels = list(results.keys())
metrics = list(results['LSTM'].keys())

fig, ax = plt.subplots(figsize=(10,6))
for metric in metrics:
    ax.plot(labels, [results[model][metric] for model in labels], marker='o', label=metric)
ax.set_title('Model Comparison: LSTM vs GNN vs Hybrid')
ax.set_ylabel('Score')
ax.legend()
plt.savefig('model_comparison_results.png')
plt.show()

# Save results to CSV
pd.DataFrame(results).T.to_csv('model_comparison_results.csv')

print('Evaluation complete. Results saved as model_comparison_results.png and model_comparison_results.csv')
