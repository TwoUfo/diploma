import torch
import torch.nn as nn

from src.models.mtl_model import mdn_nll


class MultiTaskLoss(nn.Module):
    """Four-task loss with Kendall uncertainty weighting.

    * Class — cross-entropy with optional inverse-frequency weights.
    * Diameter, albedo — masked MSE in log-space. The albedo head already
      adds the class-conditional prior bias, so we still compare to the
      full log(albedo) target.
    * Rotation period — masked Mixture Density Network NLL. Kendall scaling
      still wraps it so the task's overall contribution is auto-balanced
      with the other three; the σ_k *inside* the MDN model the per-mode
      noise.
    """

    def __init__(self, n_classes: int, class_weights: torch.Tensor = None):
        super().__init__()
        self.log_sigma_class = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_diam = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_albedo = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_rot = nn.Parameter(torch.tensor(0.0))

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
        pred_diam: torch.Tensor,
        pred_albedo: torch.Tensor,
        rot_log_pi: torch.Tensor,
        rot_mu: torch.Tensor,
        rot_log_sigma: torch.Tensor,
        true_class: torch.Tensor,
        true_diam: torch.Tensor,
        true_albedo: torch.Tensor,
        true_rot: torch.Tensor,
        mask_diam: torch.Tensor,
        mask_albedo: torch.Tensor,
        mask_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_class = self.class_loss_fn(pred_class, true_class)
        loss_diam = self._masked_mse(pred_diam, true_diam, mask_diam)
        loss_albedo = self._masked_mse(pred_albedo, true_albedo, mask_albedo)
        loss_rot = mdn_nll(rot_log_pi, rot_mu, rot_log_sigma, true_rot, mask_rot)

        sigma_class = torch.exp(self.log_sigma_class)
        sigma_diam = torch.exp(self.log_sigma_diam)
        sigma_albedo = torch.exp(self.log_sigma_albedo)
        sigma_rot = torch.exp(self.log_sigma_rot)

        total = (
            loss_class / (2 * sigma_class**2) + self.log_sigma_class
            + loss_diam / (2 * sigma_diam**2) + self.log_sigma_diam
            + loss_albedo / (2 * sigma_albedo**2) + self.log_sigma_albedo
            + loss_rot / (2 * sigma_rot**2) + self.log_sigma_rot
        )

        metrics = {
            "loss_total": total.item(),
            "loss_class": loss_class.item(),
            "loss_diam": loss_diam.item(),
            "loss_albedo": loss_albedo.item(),
            "loss_rot": loss_rot.item(),
            "sigma_class": sigma_class.item(),
            "sigma_diam": sigma_diam.item(),
            "sigma_albedo": sigma_albedo.item(),
            "sigma_rot": sigma_rot.item(),
        }

        return total, metrics
