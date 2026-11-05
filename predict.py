"""Run a trained checkpoint on your own data.

  python predict.py options --checkpoint checkpoints/options/best_options.pt --data my_options.csv
  python predict.py bl      --checkpoint checkpoints/bl/best_bl.pt           --data my_portfolio.csv
"""
import argparse
import sys

from lstm_bl.dataset import DataValidationError as BLDataError
from lstm_options.dataset import DataValidationError as OptionsDataError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict with a trained LSTM checkpoint")
    sub = parser.add_subparsers(dest="model", required=True)
    p = sub.add_parser("options", help="price each row of an options history")
    p.add_argument("--checkpoint", default="checkpoints/options/best_options.pt")
    p.add_argument("--data", required=True, help="CSV / .xlsx / .json in the options schema")
    p.add_argument("--option_type", choices=["call", "put"], default=None,
                   help="default for rows without an option_type column")
    p.add_argument("--out", help="write predictions to this CSV")
    p = sub.add_parser("bl", help="portfolio weights for the latest window")
    p.add_argument("--checkpoint", default="checkpoints/bl/best_bl.pt")
    p.add_argument("--data", required=True, help="CSV / .xlsx / .json in the portfolio schema")
    p.add_argument("--prices", action="store_true", help="asset columns are prices, not returns")
    p.add_argument("--out", help="write weights to this CSV")
    args = parser.parse_args(argv)

    try:
        if args.model == "options":
            from lstm_options.predict import predict_options
            df = predict_options(args.checkpoint, args.data, option_type=args.option_type)
        else:
            from lstm_bl.predict import predict_bl_weights
            df = predict_bl_weights(args.checkpoint, args.data, prices=args.prices)
    except (BLDataError, OptionsDataError) as e:
        print(str(e), file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e.filename}. Train first (run_options.py / run_bl.py).", file=sys.stderr)
        return 1
    if args.model == "bl" and df.attrs.get("as_of"):
        print(f"Weights as of {df.attrs['as_of']}:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
