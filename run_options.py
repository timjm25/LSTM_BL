"""Entry point for training the LSTM Options Pricing model."""
import argparse

from lstm_options.config import OptionsModelConfig, OptionsTrainConfig
from lstm_options.dataset import DataValidationError
from lstm_options.train import OptionsTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Options Pricing model")
    parser.add_argument("--data", default=None, help="your options file (.csv/.xlsx/.json); overrides --data_dir")
    parser.add_argument("--data_dir", default="data/options", help="first data file found here is used")
    parser.add_argument("--synthetic", action="store_true", help="train on synthetic demo data")
    parser.add_argument("--synthetic_paths", type=int, default=1000)
    parser.add_argument("--bsm_residual", action="store_true",
                        help="learn the correction to the BSM price rather than the raw price")
    parser.add_argument("--checkpoint_dir", default="checkpoints/options")
    parser.add_argument("--num_workers", type=int, default=2)
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
        data_file=args.data,
        synthetic=args.synthetic,
        synthetic_paths=args.synthetic_paths,
        bsm_residual=args.bsm_residual,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        warmup_epochs=min(5, max(0, args.epochs - 1)),
    )

    try:
        trainer = OptionsTrainer(train_cfg, model_cfg, option_type=args.option_type)
    except DataValidationError as e:
        raise SystemExit(str(e))
    history = trainer.run()
    print(f"\nBest val loss: {min(history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
