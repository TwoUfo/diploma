import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes: int, class_weights: torch.Tensor = None):
        super().__init__()
        self.log_sigma_class = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_h = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_diam = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_albedo = nn.Parameter(torch.tensor(0.0))

        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.reg_loss_fn = nn.MSELoss(reduction="none")

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.reg_loss_fn(pred, target)
        if mask.sum() > 0:
            return (raw * mask).sum() / mask.sum()
        return torch.tensor(0.0, device=pred.device)

    def forward(
        self,
        pred_class: torch.Tensor,
        pred_h: torch.Tensor,
        pred_diam: torch.Tensor,
        pred_albedo: torch.Tensor,
        true_class: torch.Tensor,
        true_h: torch.Tensor,
        true_diam: torch.Tensor,
        true_albedo: torch.Tensor,
        mask_h: torch.Tensor,
        mask_diam: torch.Tensor,
        mask_albedo: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_class = self.class_loss_fn(pred_class, true_class)
        loss_h = self._masked_mse(pred_h, true_h, mask_h)
        loss_diam = self._masked_mse(pred_diam, true_diam, mask_diam)
        loss_albedo = self._masked_mse(pred_albedo, true_albedo, mask_albedo)

        sigma_class = torch.exp(self.log_sigma_class)
        sigma_h = torch.exp(self.log_sigma_h)
        sigma_diam = torch.exp(self.log_sigma_diam)
        sigma_albedo = torch.exp(self.log_sigma_albedo)

        total = (
            loss_class / (2 * sigma_class**2) + self.log_sigma_class
            + loss_h / (2 * sigma_h**2) + self.log_sigma_h
            + loss_diam / (2 * sigma_diam**2) + self.log_sigma_diam
            + loss_albedo / (2 * sigma_albedo**2) + self.log_sigma_albedo
        )

        metrics = {
            "loss_total": total.item(),
            "loss_class": loss_class.item(),
            "loss_h": loss_h.item(),
            "loss_diam": loss_diam.item(),
            "loss_albedo": loss_albedo.item(),
            "sigma_class": sigma_class.item(),
            "sigma_h": sigma_h.item(),
            "sigma_diam": sigma_diam.item(),
            "sigma_albedo": sigma_albedo.item(),
        }

        return total, metrics
