"""Configuration dataclasses for options pricing LSTM."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class OptionsModelConfig:
    # Input features: [S, K, T, r, sigma, moneyness, delta, gamma, vega, theta]
    input_dim: int = 10
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    output_dim: int = 1          # option price
    sequence_length: int = 30    # lookback window (trading days)


@dataclass
class OptionsTrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"    # cosine | step | none
    warmup_epochs: int = 5
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42
    checkpoint_dir: str = "checkpoints/options"
    data_dir: str = "data/options"
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "auto"
