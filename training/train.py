"""
Training loop for Hybrid LSTM + Transformer-XL.
"""

import torch
from torch.optim import Adam
from .utils import loss_fn, MAE, RMSE

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cpu"):
    optimizer = Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, Y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                out = model(X)
                loss = loss_fn(out, Y)
                val_losses.append(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {sum(train_losses)/len(train_losses):.4f}, "
              f"Val Loss: {sum(val_losses)/len(val_losses):.4f}")
