from dataclasses import dataclass, field
import yaml


@dataclass
class DataConfig:
    raw_path: str = "data/raw/dataset.csv"
    processed_dir: str = "data/processed"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


@dataclass
class ModelConfig:
    backbone_layers: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    head_hidden: int = 32
    dropout_shared: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])
    dropout_head: float = 0.1
    n_classes: int = 10


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_grad_norm: float = 1.0
    min_lr: float = 1e-5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})) if "data" in raw else DataConfig(),
            model=ModelConfig(**raw.get("model", {})) if "model" in raw else ModelConfig(),
            training=TrainingConfig(**raw.get("training", {})) if "training" in raw else TrainingConfig(),
        )
