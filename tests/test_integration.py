import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.dataset import AsteroidDataset
from src.data.preprocessing import build_splits


def _make_raw_csv(path, per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    classes = ["MBA", "APO", "TNO"]
    n = per_class * len(classes)
    df = pd.DataFrame({
        "class": np.repeat(classes, per_class),
        "H": rng.uniform(12.0, 22.0, n),
        "diameter": rng.uniform(0.5, 50.0, n),
        "albedo": rng.uniform(0.03, 0.4, n),
        "rot_per": rng.uniform(2.5, 40.0, n),
        "neo": rng.choice(["Y", "N"], n),
        "pha": rng.choice(["Y", "N"], n),
        "a": rng.uniform(1.0, 40.0, n),
        "e": rng.uniform(0.0, 0.6, n),
        "i": rng.uniform(0.0, 30.0, n),
        "ma": rng.uniform(0.0, 360.0, n),
        "moid": rng.uniform(0.01, 1.0, n),
        "rms": rng.uniform(0.1, 0.6, n),
        "sigma_a": rng.uniform(1e-4, 1e-2, n),
        "q": rng.uniform(0.7, 3.0, n),  # leak-колонка: має бути викинута
    })
    df.to_csv(path, index=False)


class PreprocessingToDatasetIntegrationTests(unittest.TestCase):
    def test_build_splits_output_matches_dataset_and_scaler_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw_csv = tmp / "raw.csv"
            _make_raw_csv(raw_csv)

            build_splits(str(raw_csv), str(tmp), test_size=0.2, val_size=0.2)

            scaler = joblib.load(tmp / "scaler.joblib")
            train_df = pd.read_parquet(tmp / "train.parquet")

            # Leak-колонки й сирі цілі не повинні потрапити у вибірку.
            for leaked in ("q", "albedo", "diameter", "rot_per"):
                self.assertNotIn(leaked, train_df.columns)

            # Маски наявні у схемі.
            for mask_col in ("diameter_mask", "albedo_mask", "rot_mask"):
                self.assertIn(mask_col, train_df.columns)

            # Контракт «schema ↔ Dataset ↔ scaler».
            dataset = AsteroidDataset(tmp / "train.parquet")
            self.assertEqual(dataset.n_features, scaler.n_features_in_)
            self.assertEqual(dataset.n_features, len(scaler.feature_names_in_))

            item = dataset[0]
            self.assertEqual(
                set(item),
                {
                    "features", "class_label",
                    "diameter_target", "albedo_target", "rot_target",
                    "diameter_mask", "albedo_mask", "rot_mask",
                },
            )
            self.assertEqual(item["features"].shape[0], dataset.n_features)


if __name__ == "__main__":
    unittest.main()
