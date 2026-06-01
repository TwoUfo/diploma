import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import AsteroidDataset
from src.models.losses import MultiTaskLoss
from src.models.mtl_model import AsteroidMTLModel
from src.training.trainer import Trainer


def _make_model_and_criterion(n_features=6, n_classes=3):
    model = AsteroidMTLModel(
        n_features=n_features,
        n_classes=n_classes,
        backbone_layers=[8],
        backbone_dropouts=[0.0],
        head_hidden=4,
        head_dropout=0.0,
        rot_mdn_components=3,
    )
    criterion = MultiTaskLoss(n_classes=n_classes)
    return model, criterion


def _write_processed_parquet(path, n_rows, n_features=6, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"feat_{j}": rng.normal(size=n_rows) for j in range(n_features)}
    data["class_label"] = rng.integers(0, n_classes, size=n_rows)
    data["diameter_target"] = rng.normal(size=n_rows)
    data["albedo_target"] = rng.normal(size=n_rows)
    data["rot_target"] = rng.normal(size=n_rows)
    data["diameter_mask"] = np.ones(n_rows)
    data["albedo_mask"] = np.ones(n_rows)
    data["rot_mask"] = (rng.random(n_rows) > 0.5).astype(float)
    pd.DataFrame(data).to_parquet(path, index=False)


class TrainerTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_learned_uncertainties(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, criterion = _make_model_and_criterion()
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(criterion.parameters())
            )
            trainer = Trainer(model, criterion, optimizer, checkpoint_dir=tmp)

            # Імітуємо навчені невизначеності, змінивши параметри criterion.
            with torch.no_grad():
                for p in criterion.parameters():
                    p.add_(0.5)
            saved_state = {k: v.clone() for k, v in criterion.state_dict().items()}

            trainer.save_checkpoint("ckpt.pt")

            # Свіжі об'єкти зі стандартною ініціалізацією.
            model2, criterion2 = _make_model_and_criterion()
            optimizer2 = torch.optim.AdamW(
                list(model2.parameters()) + list(criterion2.parameters())
            )
            trainer2 = Trainer(model2, criterion2, optimizer2, checkpoint_dir=tmp)
            trainer2.load_checkpoint("ckpt.pt")

            for key, value in saved_state.items():
                self.assertTrue(
                    torch.allclose(criterion2.state_dict()[key], value),
                    msg=f"невідновлений параметр criterion: {key}",
                )

    def test_fit_runs_end_to_end_on_synthetic_processed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_path = tmp / "train.parquet"
            val_path = tmp / "val.parquet"
            _write_processed_parquet(train_path, n_rows=48, seed=1)
            _write_processed_parquet(val_path, n_rows=16, seed=2)

            train_loader = DataLoader(AsteroidDataset(train_path), batch_size=8, shuffle=True)
            val_loader = DataLoader(AsteroidDataset(val_path), batch_size=8)

            model, criterion = _make_model_and_criterion()
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(criterion.parameters())
            )
            trainer = Trainer(model, criterion, optimizer, checkpoint_dir=tmp)

            history = trainer.fit(train_loader, val_loader, epochs=2, patience=5)

            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(history[-1]["val_loss"]))
            self.assertTrue((tmp / "mtl_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
