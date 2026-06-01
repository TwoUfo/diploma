import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import AsteroidDataset


class AsteroidDatasetTests(unittest.TestCase):
    def test_dataset_splits_features_targets_and_masks_by_column_names(self):
        df = pd.DataFrame(
            {
                "neo": [0.0, 1.0],
                "H": [12.3, 18.4],
                "tisserand_j": [3.1, 2.8],
                "class_label": [4, 1],
                "diameter_target": [1.2, 0.0],
                "albedo_target": [-2.3, 0.0],
                "rot_target": [2.1, 0.0],
                "diameter_mask": [1.0, 0.0],
                "albedo_mask": [1.0, 0.0],
                "rot_mask": [1.0, 0.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.parquet"
            df.to_parquet(path, index=False)

            ds = AsteroidDataset(str(path))

        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.n_features, 3)
        self.assertEqual(ds.features.shape, (2, 3))
        self.assertEqual(ds.class_labels.dtype, torch.long)
        self.assertEqual(ds.diameter_targets.dtype, torch.float32)

        item = ds[0]
        self.assertEqual(
            set(item),
            {
                "features",
                "class_label",
                "diameter_target",
                "albedo_target",
                "rot_target",
                "diameter_mask",
                "albedo_mask",
                "rot_mask",
            },
        )
        self.assertTrue(torch.equal(item["features"], torch.tensor([0.0, 12.3, 3.1])))


if __name__ == "__main__":
    unittest.main()
