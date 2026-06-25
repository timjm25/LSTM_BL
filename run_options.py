"""Entry point for training the LSTM Options Pricing model."""
import argparse

from lstm_options.config import OptionsModelConfig, OptionsTrainConfig
from lstm_options.train import OptionsTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Options Pricing model")
    parser.add_argument("--data_dir", default="data/options")
    parser.add_argument("--option_type", default="call", choices=["call", "put"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_cfg = OptionsModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sequence_length=args.seq_len,
    )
    train_cfg = OptionsTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        device=args.device,
    )

    trainer = OptionsTrainer(train_cfg, model_cfg, option_type=args.option_type)
    history = trainer.run()
    print(f"\nBest val loss: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
