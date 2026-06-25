"""Configuration for LSTM Black-Litterman model."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BLModelConfig:
    n_assets: int = 10              # number of assets in the portfolio
    macro_features: int = 8         # macro/sentiment features per timestep
    total_input_dim: int = 18       # n_assets + macro_features per timestep (returns + macro)
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    sequence_length: int = 60       # lookback window (trading days, ~3 months)
    tau: float = 0.05               # BL uncertainty scaling factor


@dataclass
class BLTrainConfig:
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42
    checkpoint_dir: str = "checkpoints/bl"
    data_dir: str = "data/bl"
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "auto"
