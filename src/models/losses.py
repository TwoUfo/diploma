import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, n_classes: int, class_weights: torch.Tensor = None):
        super().__init__()
        self.log_sigma_class = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_h = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_diam = nn.Parameter(torch.tensor(0.0))

        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.reg_loss_fn = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred_class: torch.Tensor,
        pred_h: torch.Tensor,
        pred_diam: torch.Tensor,
        true_class: torch.Tensor,
        true_h: torch.Tensor,
        true_diam: torch.Tensor,
        mask_h: torch.Tensor,
        mask_diam: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_class = self.class_loss_fn(pred_class, true_class)

        loss_h_raw = self.reg_loss_fn(pred_h, true_h)
        if mask_h.sum() > 0:
            loss_h = (loss_h_raw * mask_h).sum() / mask_h.sum()
        else:
            loss_h = torch.tensor(0.0, device=pred_h.device)

        loss_diam_raw = self.reg_loss_fn(pred_diam, true_diam)
        if mask_diam.sum() > 0:
            loss_diam = (loss_diam_raw * mask_diam).sum() / mask_diam.sum()
        else:
            loss_diam = torch.tensor(0.0, device=pred_diam.device)

        sigma_class = torch.exp(self.log_sigma_class)
        sigma_h = torch.exp(self.log_sigma_h)
        sigma_diam = torch.exp(self.log_sigma_diam)

        total = (
            loss_class / (2 * sigma_class**2) + self.log_sigma_class
            + loss_h / (2 * sigma_h**2) + self.log_sigma_h
            + loss_diam / (2 * sigma_diam**2) + self.log_sigma_diam
        )

        metrics = {
            "loss_total": total.item(),
            "loss_class": loss_class.item(),
            "loss_h": loss_h.item(),
            "loss_diam": loss_diam.item(),
            "sigma_class": sigma_class.item(),
            "sigma_h": sigma_h.item(),
            "sigma_diam": sigma_diam.item(),
        }

        return total, metrics
