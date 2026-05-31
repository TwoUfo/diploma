import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        layer_sizes: list[int] = None,
        dropouts: list[float] = None,
    ):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [512, 256, 128, 64]
        if dropouts is None:
            dropouts = [0.3, 0.3, 0.2, 0.2]

        layers = []
        in_size = n_features
        for size, drop in zip(layer_sizes, dropouts):
            layers.extend([
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(drop),
            ])
            in_size = size

        self.network = nn.Sequential(*layers)
        self.output_size = layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TaskHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MDNHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, K: int = 5, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pi_layer = nn.Linear(hidden_size, K)
        self.mu_layer = nn.Linear(hidden_size, K)
        self.log_sigma_layer = nn.Linear(hidden_size, K)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        log_pi = F.log_softmax(self.pi_layer(h), dim=-1)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h).clamp(-5.0, 3.0)
        return log_pi, mu, log_sigma


def mdn_map(log_pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    most_likely = log_pi.argmax(dim=-1, keepdim=True)
    return mu.gather(-1, most_likely).squeeze(-1)


def mdn_nll(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    target = target.unsqueeze(-1)
    log_prob_k = (
        -0.5 * ((target - mu) / log_sigma.exp()) ** 2
        - log_sigma
        - 0.5 * math.log(2 * math.pi)
    )
    log_lik = torch.logsumexp(log_pi + log_prob_k, dim=-1)
    nll = -log_lik
    if mask.sum() > 0:
        return (nll * mask).sum() / mask.sum()
    return torch.tensor(0.0, device=nll.device)


class AsteroidMTLModel(nn.Module):
    """
    4-task MTL: class, diameter, albedo, rotation period.

    Albedo head: predicts the *residual* against a class-conditional prior.
    The prior is registered as a buffer (computed from train data in
    preprocessing) and added to the residual at output time.
    Rotation head: Mixture Density Network with ``K`` Gaussian components
    so the head can model the bimodal YORP-spun vs primordial distribution
    instead of regressing to the population mean.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 10,
        backbone_layers: list[int] = None,
        backbone_dropouts: list[float] = None,
        head_hidden: int = 32,
        head_dropout: float = 0.1,
        class_albedo_prior: torch.Tensor = None,
        rot_mdn_components: int = 5,
    ):
        super().__init__()
        self.backbone = SharedBackbone(n_features, backbone_layers, backbone_dropouts)
        shared_dim = self.backbone.output_size

        self.class_head = TaskHead(shared_dim, head_hidden, n_classes, head_dropout)
        self.diameter_head = TaskHead(shared_dim, head_hidden, 1, head_dropout)
        self.albedo_residual_head = TaskHead(shared_dim, head_hidden, 1, head_dropout)
        self.rot_head = MDNHead(shared_dim, head_hidden, K=rot_mdn_components, dropout=head_dropout)

        if class_albedo_prior is None:
            class_albedo_prior = torch.full((n_classes,), float(math.log(0.1)))
        elif not isinstance(class_albedo_prior, torch.Tensor):
            class_albedo_prior = torch.as_tensor(class_albedo_prior, dtype=torch.float32)
        self.register_buffer("class_albedo_prior", class_albedo_prior.float())

    def forward(self, x: torch.Tensor, class_label: torch.Tensor = None) -> dict[str, torch.Tensor]:
        shared = self.backbone(x)
        class_logits = self.class_head(shared)

        albedo_residual = self.albedo_residual_head(shared).squeeze(-1)
        if class_label is not None:
            albedo_bias = self.class_albedo_prior[class_label]
        else:
            class_probs = F.softmax(class_logits, dim=-1).detach()
            albedo_bias = (class_probs * self.class_albedo_prior).sum(dim=-1)
        albedo_pred = albedo_residual + albedo_bias

        rot_log_pi, rot_mu, rot_log_sigma = self.rot_head(shared)
        rot_pred = mdn_map(rot_log_pi, rot_mu)

        return {
            "class_logits": class_logits,
            "diameter_pred": self.diameter_head(shared).squeeze(-1),
            "albedo_pred": albedo_pred,
            "rot_pred": rot_pred,
            "rot_log_pi": rot_log_pi,
            "rot_mu": rot_mu,
            "rot_log_sigma": rot_log_sigma,
            "shared_repr": shared,
        }
