"""Backward-compatible entrypoint for the RUL experiment runner."""

import argparse

from runner.rul_runner import RULDataConfig, RULRunConfig, main


def parse_args():
    parser = argparse.ArgumentParser(description="Run the backward-compatible RUL experiment entrypoint.")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--condition", default="35Hz12kN")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default = RULRunConfig(data=RULDataConfig(root_dir=args.root_dir, condition=args.condition))
    main(default)
