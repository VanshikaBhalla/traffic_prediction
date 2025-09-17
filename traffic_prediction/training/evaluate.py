"""
Evaluation on test set.
"""

import torch
from .utils import MAE, RMSE

def evaluate_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    mae_list, rmse_list = [], []
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            mae_list.append(MAE(pred, Y))
            rmse_list.append(RMSE(pred, Y))

    print(f"Test MAE: {sum(mae_list)/len(mae_list):.4f}, RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
