import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


file_path = "stock_data.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

stock_cols = df.columns.drop('Date')
stock_data = df[stock_cols]


window = 50
features = stock_data[-window:].T.values 

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

x = torch.tensor(features_scaled, dtype=torch.float)


corr_matrix = stock_data.corr().values
threshold = 0.6  

edges = np.array(np.where(np.abs(corr_matrix) > threshold))
edges = edges[:, edges[0] != edges[1]]

edge_index = torch.tensor(edges, dtype=torch.long)


horizon = 5
future = stock_data.shift(-horizon)
labels = ((future - stock_data) > 0).astype(int)
label_vec = labels.iloc[-window].values  

y = label_vec.astype(float)

train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

y = torch.tensor(y, dtype=torch.float)


data = Data(x=x, edge_index=edge_index, y=y)


class GNNStockPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1) 
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()

model = GNNStockPredictor(in_channels=data.num_node_features, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.BCELoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(idx):
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = (out[idx] > 0.5).float()
        acc = (preds == data.y[idx]).sum().item() / len(idx)
    return acc

num_epochs = 100
for epoch in range(num_epochs):
    loss = train()
    if epoch % 10 == 0 or epoch == num_epochs-1:
        train_acc = evaluate(train_idx)
        test_acc = evaluate(test_idx)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

train_acc = evaluate(train_idx)
test_acc = evaluate(test_idx)
print(f"Final Train Accuracy: {train_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
