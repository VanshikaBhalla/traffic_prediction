"""
Utility functions: metrics and losses.
"""

import torch
import torch.nn as nn

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def RMSE(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0)
    mask = mask.float()
    loss = torch.abs(y_pred - y_true) * mask
    return torch.mean(loss)

# Simple MSE loss for training
loss_fn = nn.MSELoss()
