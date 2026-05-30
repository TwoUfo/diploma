import pandas as pd
import torch
from torch.utils.data import Dataset


class AsteroidDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)

        target_cols = [
            "class_label", "diameter_target", "albedo_target", "rot_target",
            "diameter_mask", "albedo_mask", "rot_mask",
        ]
        feature_cols = [c for c in df.columns if c not in target_cols]

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.class_labels = torch.tensor(df["class_label"].values, dtype=torch.long)
        self.diameter_targets = torch.tensor(df["diameter_target"].values, dtype=torch.float32)
        self.albedo_targets = torch.tensor(df["albedo_target"].values, dtype=torch.float32)
        self.rot_targets = torch.tensor(df["rot_target"].values, dtype=torch.float32)
        self.diameter_mask = torch.tensor(df["diameter_mask"].values, dtype=torch.float32)
        self.albedo_mask = torch.tensor(df["albedo_mask"].values, dtype=torch.float32)
        self.rot_mask = torch.tensor(df["rot_mask"].values, dtype=torch.float32)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "class_label": self.class_labels[idx],
            "diameter_target": self.diameter_targets[idx],
            "albedo_target": self.albedo_targets[idx],
            "rot_target": self.rot_targets[idx],
            "diameter_mask": self.diameter_mask[idx],
            "albedo_mask": self.albedo_mask[idx],
            "rot_mask": self.rot_mask[idx],
        }
