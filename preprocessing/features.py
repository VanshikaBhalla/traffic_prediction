"""
Generate time-based features and adjacency matrix.
"""

import numpy as np
import pandas as pd

def add_time_features(df):
    df_time = df.copy()
    df_time = df_time.reset_index().rename(columns={"index": "timestamp"})
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])

    # Basic time features
    df_time["hour"] = df_time["timestamp"].dt.hour
    df_time["dayofweek"] = df_time["timestamp"].dt.dayofweek
    df_time["is_weekend"] = df_time["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encoding
    df_time["hour_sin"] = np.sin(2 * np.pi * df_time["hour"] / 24)
    df_time["hour_cos"] = np.cos(2 * np.pi * df_time["hour"] / 24)
    df_time["dow_sin"] = np.sin(2 * np.pi * df_time["dayofweek"] / 7)
    df_time["dow_cos"] = np.cos(2 * np.pi * df_time["dayofweek"] / 7)

    # Fallback holiday flag
    df_time["is_holiday"] = 0

    return df_time

def build_adjacency_matrix(df, threshold=0.5):
    sensor_corr = df.corr().values
    adj_matrix = (np.abs(sensor_corr) >= threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix
