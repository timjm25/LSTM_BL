"""Entry point for training the LSTM Black-Litterman portfolio model."""
import argparse

from lstm_bl.config import BLModelConfig, BLTrainConfig
from lstm_bl.train import BLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Black-Litterman model")
    parser.add_argument("--data_dir", default="data/bl")
    parser.add_argument("--n_assets", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    n = args.n_assets
    model_cfg = BLModelConfig(
        n_assets=n,
        total_input_dim=n + 8,
        hidden_dim=args.hidden_dim,
        sequence_length=args.seq_len,
    )
    train_cfg = BLTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        device=args.device,
    )

    trainer = BLTrainer(train_cfg, model_cfg)
    history = trainer.run()
    print(f"\nBest val loss: {min(history['val_loss']):.6f}")
    print(f"Best val Sharpe: {max(history['val_sharpe']):.3f}")


if __name__ == "__main__":
    main()
