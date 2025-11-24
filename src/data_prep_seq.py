# src/data_prep_seq.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from .config import data_config
from .utils import get_logger

logger = get_logger(__name__)

def bin_to_grid(df, grid_size=10, features=None):
    if features is None:
        features = data_config.features
    df = df.copy()
    df['lat_bin'] = pd.cut(df['latitude'], bins=grid_size, labels=False)
    df['lon_bin'] = pd.cut(df['longitude'], bins=grid_size, labels=False)
    df = df.dropna(subset=['lat_bin', 'lon_bin'])
    df['lat_bin'] = df['lat_bin'].astype(int)
    df['lon_bin'] = df['lon_bin'].astype(int)

    grid = np.zeros((grid_size, grid_size, len(features)))
    for _, row in df.iterrows():
        i, j = int(row['lat_bin']), int(row['lon_bin'])
        if 0 <= i < grid_size and 0 <= j < grid_size:
            grid[i, j] = row[features].values
    return grid

class SpatialSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # (N, T, C, H, W)
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.float)

def create_spatial_sequences(sample_ratio=0.05):
    logger.info("Creating spatial sequences for DL models...")
    
    # === READ train.parquet WITH datetime ===
    df = pd.read_parquet(data_config.train_path)
    logger.info(f"Loaded train.parquet: {df.shape}, columns: {df.columns.tolist()}")

    n = int(len(df) * sample_ratio)
    idx = np.random.choice(len(df), n, replace=False)
    df = df.iloc[idx].copy()

    # === ENSURE datetime is parsed ===
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    daily_grids = []
    daily_labels = []
    for date, group in df.groupby('date'):
        grid = bin_to_grid(group, grid_size=data_config.grid_size)
        label = group['Wildfire'].any()
        daily_grids.append(grid)
        daily_labels.append(label)

    sequences = np.stack(daily_grids)  # (T, H, W, C)
    labels = np.array(daily_labels)

    seq_len = data_config.seq_length
    X, y = [], []
    for i in range(len(sequences) - seq_len + 1):
        X.append(sequences[i:i + seq_len])
        y.append(labels[i + seq_len - 1])
    X = np.stack(X).transpose(0, 1, 4, 2, 3)  # (N, T, C, H, W)
    y = np.array(y)

    dataset = SpatialSequenceDataset(X, y)
    return DataLoader(dataset, batch_size=data_config.batch_size, shuffle=True)