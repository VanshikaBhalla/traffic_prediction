"""
Preprocess raw traffic data:
- Load .h5 dataset
- Handle zeros (treat long consecutive zeros as missing)
- Interpolate missing values
- Normalize per sensor
"""

import pandas as pd
import numpy as np

def load_and_clean_data(file_path, zero_threshold=12):
    # Load dataset
    df = pd.read_hdf(file_path, key="df")

    # Replace long consecutive zeros with NaN
    def replace_long_zeros(series, threshold):
        values = series.values.copy()
        zero_runs = np.split(np.where(values == 0)[0],
                             np.where(np.diff(np.where(values == 0)[0]) != 1)[0] + 1)
        for run in zero_runs:
            if len(run) >= threshold:
                values[run] = np.nan
        return pd.Series(values, index=series.index)

    df_clean = df.apply(lambda col: replace_long_zeros(col, threshold=zero_threshold))

    # Interpolate + fill edges
    df_clean = df_clean.interpolate(method="linear", axis=0).ffill().bfill()

    return df_clean

def normalize_data(df):
    means = df.mean()
    stds = df.std()
    df_norm = (df - means) / stds
    return df_norm, means, stds
