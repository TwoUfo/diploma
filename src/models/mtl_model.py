import torch
import torch.nn as nn


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


class AsteroidMTLModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 10,
        backbone_layers: list[int] = None,
        backbone_dropouts: list[float] = None,
        head_hidden: int = 32,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SharedBackbone(n_features, backbone_layers, backbone_dropouts)
        shared_dim = self.backbone.output_size

        self.class_head = TaskHead(shared_dim, head_hidden, n_classes, head_dropout)
        self.h_head = TaskHead(shared_dim, head_hidden, 1, head_dropout)
        self.diameter_head = TaskHead(shared_dim, head_hidden, 1, head_dropout)
        self.albedo_head = TaskHead(shared_dim, head_hidden, 1, head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.backbone(x)
        return {
            "class_logits": self.class_head(shared),
            "h_pred": self.h_head(shared).squeeze(-1),
            "diameter_pred": self.diameter_head(shared).squeeze(-1),
            "albedo_pred": self.albedo_head(shared).squeeze(-1),
            "shared_repr": shared,
        }
