import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.mtl_model import AsteroidMTLModel
from src.models.losses import MultiTaskLoss
from src.training.metrics import classification_metrics, regression_metrics


class Trainer:
    def __init__(
        self,
        model: AsteroidMTLModel,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "models",
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> dict:
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        loss_components = {"loss_class": 0.0, "loss_diam": 0.0, "loss_albedo": 0.0, "loss_rot": 0.0}
        all_class_true, all_class_pred = [], []
        all_diam_true, all_diam_pred, all_diam_mask = [], [], []
        all_alb_true, all_alb_pred, all_alb_mask = [], [], []
        all_rot_true, all_rot_pred, all_rot_mask = [], [], []

        ctx = torch.no_grad() if not train else torch.enable_grad()
        with ctx:
            for batch in loader:
                features = batch["features"].to(self.device)
                class_labels = batch["class_label"].to(self.device)
                diam_targets = batch["diameter_target"].to(self.device)
                alb_targets = batch["albedo_target"].to(self.device)
                rot_targets = batch["rot_target"].to(self.device)
                diam_mask = batch["diameter_mask"].to(self.device)
                alb_mask = batch["albedo_mask"].to(self.device)
                rot_mask = batch["rot_mask"].to(self.device)

                outputs = self.model(features, class_label=class_labels if train else None)
                loss, metrics = self.criterion(
                    outputs["class_logits"],
                    outputs["diameter_pred"], outputs["albedo_pred"],
                    outputs["rot_log_pi"], outputs["rot_mu"], outputs["rot_log_sigma"],
                    class_labels, diam_targets, alb_targets, rot_targets,
                    diam_mask, alb_mask, rot_mask,
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.criterion.parameters()),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()

                total_loss += loss.item() * len(features)
                for k in loss_components:
                    loss_components[k] += metrics[k] * len(features)

                preds = outputs["class_logits"].argmax(dim=1).cpu().numpy()
                all_class_true.append(class_labels.cpu().numpy())
                all_class_pred.append(preds)
                all_diam_true.append(diam_targets.cpu().numpy())
                all_diam_pred.append(outputs["diameter_pred"].detach().cpu().numpy())
                all_diam_mask.append(diam_mask.cpu().numpy())
                all_alb_true.append(alb_targets.cpu().numpy())
                all_alb_pred.append(outputs["albedo_pred"].detach().cpu().numpy())
                all_alb_mask.append(alb_mask.cpu().numpy())
                all_rot_true.append(rot_targets.cpu().numpy())
                all_rot_pred.append(outputs["rot_pred"].detach().cpu().numpy())
                all_rot_mask.append(rot_mask.cpu().numpy())

        n = len(loader.dataset)
        result = {"loss": total_loss / n}
        for k in loss_components:
            result[k] = loss_components[k] / n

        y_cls_true = np.concatenate(all_class_true)
        y_cls_pred = np.concatenate(all_class_pred)
        cls_m = classification_metrics(y_cls_true, y_cls_pred)
        result.update({f"class_{k}": v for k, v in cls_m.items()})

        y_d_true = np.concatenate(all_diam_true)
        y_d_pred = np.concatenate(all_diam_pred)
        d_m_arr = np.concatenate(all_diam_mask)
        d_m = regression_metrics(y_d_true, y_d_pred, d_m_arr)
        result.update({f"diam_{k}": v for k, v in d_m.items()})

        y_a_true = np.concatenate(all_alb_true)
        y_a_pred = np.concatenate(all_alb_pred)
        a_m_arr = np.concatenate(all_alb_mask)
        a_m = regression_metrics(y_a_true, y_a_pred, a_m_arr)
        result.update({f"albedo_{k}": v for k, v in a_m.items()})

        y_r_true = np.concatenate(all_rot_true)
        y_r_pred = np.concatenate(all_rot_pred)
        r_m_arr = np.concatenate(all_rot_mask)
        r_m = regression_metrics(y_r_true, y_r_pred, r_m_arr)
        result.update({f"rot_{k}": v for k, v in r_m.items()})

        return result

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._run_epoch(loader, train=True)

    def validate(self, loader: DataLoader) -> dict:
        return self._run_epoch(loader, train=False)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ) -> list[dict]:
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {"epoch": epoch}
            record.update({f"train_{k}": v for k, v in train_metrics.items()})
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(record)

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['class_accuracy']:.4f} | "
                f"Val Diam R²: {val_metrics['diam_r2']:.4f} | "
                f"Val Alb R²: {val_metrics['albedo_r2']:.4f} | "
                f"Val Rot R²: {val_metrics['rot_r2']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
                self.save_checkpoint("mtl_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.history

    def save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
