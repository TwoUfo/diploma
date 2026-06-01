import unittest
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models.mtl_model import AsteroidMTLModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactSmokeTests(unittest.TestCase):
    def test_committed_model_artifacts_load_and_run_one_prediction(self):
        model_path = PROJECT_ROOT / "models" / "mtl_model.pt"
        scaler_path = PROJECT_ROOT / "data" / "processed" / "scaler.joblib"
        label_encoder_path = PROJECT_ROOT / "data" / "processed" / "label_encoder.joblib"
        prior_path = PROJECT_ROOT / "data" / "processed" / "class_albedo_prior.joblib"
        train_path = PROJECT_ROOT / "data" / "processed" / "train.parquet"

        required = [model_path, scaler_path, label_encoder_path, prior_path, train_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            self.skipTest(f"missing demo artifacts: {missing}")

        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        class_albedo_prior = joblib.load(prior_path)

        model = AsteroidMTLModel(
            n_features=scaler.n_features_in_,
            n_classes=len(label_encoder.classes_),
            backbone_layers=[512, 256, 128, 64],
            backbone_dropouts=[0.3, 0.3, 0.2, 0.2],
            head_hidden=32,
            head_dropout=0.1,
            class_albedo_prior=class_albedo_prior,
            rot_mdn_components=5,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        row = pd.read_parquet(train_path, columns=list(scaler.feature_names_in_)).iloc[[0]]
        x = torch.tensor(row.values, dtype=torch.float32)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(out["class_logits"].shape, (1, len(label_encoder.classes_)))
        self.assertTrue(torch.isfinite(out["diameter_pred"]).all())
        self.assertTrue(torch.isfinite(out["albedo_pred"]).all())
        self.assertTrue(torch.isfinite(out["rot_pred"]).all())


if __name__ == "__main__":
    unittest.main()
