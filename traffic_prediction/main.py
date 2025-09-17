"""

import h5py
import numpy as np
import pandas as pd

# Open the HDF5 file
file_path = "metr-la.h5"
h5_file = h5py.File(file_path, "r")

# List all datasets in the file
print("Keys:", list(h5_file.keys()))

# Read the HDF5 dataset using pandas
df = pd.read_hdf(file_path, key='df')

# Show first few rows
print(df.head(10))

print(df.isna().any().any())  # True if there is at least one NaN

file_path2 = "pems-bay.h5"
pemsh5_file = h5py.File(file_path2,'r')

print("Keys:", list(pemsh5_file.keys()))

dfpems = pd.read_hdf(file_path2, key='speed')

print(dfpems.head(10))
"""


"""
Main pipeline: preprocessing → dataset → model → training → evaluation
"""

import torch
from preprocessing.preprocess import load_and_clean_data, normalize_data
from preprocessing.features import add_time_features, build_adjacency_matrix
from preprocessing.windowing import create_windowed_dataset
from dataset.traffic_dataset import get_dataloaders
from models.hybrid_model import HybridModel
from training.train import train_model
from training.evaluate import evaluate_model

def main():
    # 1. Load + preprocess
    df_clean = load_and_clean_data("data/metr-la.h5")
    df_norm, mean, std = normalize_data(df_clean)
    df_time = add_time_features(df_norm)
    adj = build_adjacency_matrix(df_clean)

    # 2. Create dataset
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_time)
    train_loader, val_loader, test_loader = get_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    # 3. Build model
    model = HybridModel(input_dim=X_train.shape[-1], output_dim=Y_train.shape[-1])

    # 4. Train
    train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu")

    # 5. Evaluate
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
