"""
PyTorch Dataset + DataLoader for traffic forecasting.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=64):
    train_ds = TrafficDataset(X_train, Y_train)
    val_ds   = TrafficDataset(X_val, Y_val)
    test_ds  = TrafficDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
