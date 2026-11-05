"""Entry point for training the LSTM Black-Litterman portfolio model."""
import argparse

from lstm_bl.config import BLModelConfig, BLTrainConfig
from lstm_bl.dataset import DataValidationError
from lstm_bl.train import BLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Black-Litterman model")
    parser.add_argument("--data", default=None, help="your portfolio file (.csv/.xlsx/.json); overrides --data_dir")
    parser.add_argument("--data_dir", default="data/bl", help="first data file found here is used")
    parser.add_argument("--assets", default=None,
                        help="comma-separated asset columns (default: columns starting with asset_)")
    parser.add_argument("--macro", default=None,
                        help="comma-separated macro columns (default: every other non-date column)")
    parser.add_argument("--prices", action="store_true", help="asset columns are prices, not returns")
    parser.add_argument("--synthetic", action="store_true", help="train on synthetic demo data")
    parser.add_argument("--synthetic_series", type=int, default=200)
    parser.add_argument("--n_assets", type=int, default=10, help="synthetic data only; files set this")
    parser.add_argument("--checkpoint_dir", default="checkpoints/bl")
    parser.add_argument("--num_workers", type=int, default=2)
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
        data_file=args.data,
        asset_cols=[c.strip() for c in args.assets.split(",")] if args.assets else None,
        macro_cols=[c.strip() for c in args.macro.split(",") if c.strip()] if args.macro is not None else None,
        prices=args.prices,
        synthetic=args.synthetic,
        synthetic_series=args.synthetic_series,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        warmup_epochs=min(10, max(0, args.epochs - 1)),
    )

    try:
        trainer = BLTrainer(train_cfg, model_cfg)
    except DataValidationError as e:
        raise SystemExit(str(e))
    history = trainer.run()
    print(f"\nBest val loss: {min(history['val_loss']):.6f}")
    print(f"Best val Sharpe: {max(history['val_sharpe']):.3f}")


if __name__ == "__main__":
    main()
