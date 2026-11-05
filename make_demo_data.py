"""Write synthetic demo files in the real-data schema, to try the file-import path end to end.

  python make_demo_data.py            # -> data/demo/options_demo.csv, data/demo/bl_demo.csv
"""
import argparse
from pathlib import Path

from lstm_bl.dataset import write_demo_bl_csv
from lstm_options.dataset import write_demo_options_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dir", default="data/demo")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Wrote", write_demo_options_csv(out / "options_demo.csv"))
    print("Wrote", write_demo_bl_csv(out / "bl_demo.csv"))


if __name__ == "__main__":
    main()
