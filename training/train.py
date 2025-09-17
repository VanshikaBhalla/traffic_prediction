"""
Training loop for Hybrid LSTM + Transformer-XL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu", patience=5, save_path="best_model.pth"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_losses = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                out = model(X)
                loss = criterion(out, Y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)  # Save best model
            print(f"  ✅ Best model saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    # Load best model before returning
    model.load_state_dict(torch.load(save_path))
    return model
