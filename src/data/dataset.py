import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DelayDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        indices_path,
        look_back_window_size,
        forecast_horizon_size,
        feature_set,
    ):
        df = pd.read_csv(dataset_path, dtype={"lost": "boolean"})

        self.target = df["delay-rtt"].astype(float).values.astype(np.float32)
        self.data = df[feature_set].astype(float).values.astype(np.float32)
        self.indices = pd.read_csv(indices_path).values
        self.look_back_window_size = look_back_window_size
        self.forecast_horizon_size = forecast_horizon_size
        self.feature_set = feature_set

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        look_back_start = self.indices[index].item() - self.look_back_window_size + 1
        look_back_end = look_back_start + self.look_back_window_size
        forecast_horizon_end = look_back_end + self.forecast_horizon_size

        look_back_window = self.data[look_back_start:look_back_end]
        forecast_horizon = self.target[look_back_end:forecast_horizon_end]

        x = torch.tensor(look_back_window, dtype=torch.float32)
        y = torch.tensor(forecast_horizon, dtype=torch.float32)

        if torch.isnan(x).any() or torch.isnan(y).any():
            raise ValueError(
                f"NaN found in window for dataset index {index} "
                f"(Rows {look_back_start} to {forecast_horizon_end}). "
            )

        return x, y
