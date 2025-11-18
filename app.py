# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv



class GNNStockPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)  # binary output
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()


def prepare_graph(df, window=30, horizon=5, threshold=0.6):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    stock_cols = df.columns.drop('Date')
    stock_data = df[stock_cols]

    features = stock_data[-window:].T.values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float)

    corr_matrix = stock_data.corr().values
    edges = np.array(np.where(np.abs(corr_matrix) > threshold))
    edges = edges[:, edges[0] != edges[1]]
    edge_index = torch.tensor(edges, dtype=torch.long)

    future = stock_data.shift(-horizon)
    labels = ((future - stock_data) > 0).astype(int)
    label_vec = labels.iloc[-window].values
    y = torch.tensor(label_vec.astype(float), dtype=torch.float)


    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, train_idx, test_idx


# Streamlit

st.title("📈 Stock Price Trend Prediction with GNN")

uploaded_file = st.file_uploader("Upload your stock_data.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    data, train_idx, test_idx = prepare_graph(df)
    st.write(f"Graph built with **{data.num_nodes} nodes** and **{data.num_edges} edges**")

    if st.button("🚀 Train Model"):
        model = GNNStockPredictor(in_channels=data.num_node_features, hidden_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCELoss()

        progress = st.progress(0)
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                progress.progress((epoch+1)/num_epochs)
                st.write(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            preds = (model(data) > 0.5).float()
            train_acc = (preds[train_idx] == data.y[train_idx]).sum().item() / len(train_idx)
            test_acc = (preds[test_idx] == data.y[test_idx]).sum().item() / len(test_idx)

        st.success("✅ Training complete!")
        st.write(f"**Final Train Accuracy:** {train_acc:.4f}")
        st.write(f"**Final Test Accuracy:** {test_acc:.4f}")

        st.write("### Predictions per Stock Node")
        results_df = pd.DataFrame({
            "Stock": df.columns.drop("Date"),
            "Predicted Trend (1=Up, 0=Down)": preds.numpy()
        })
        st.dataframe(results_df)
