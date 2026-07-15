"""Command-line interface for auditable leaderboard tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from torch_like.automl.config import load_config
from torch_like.automl.task import LeaderboardTask


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(prog="torch-like", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    show = commands.add_parser("show")
    show.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    if args.command == "show":
        leaderboard_path = args.run_dir / "leaderboard.csv"
        if not leaderboard_path.is_file():
            parser.error(f"Leaderboard does not exist: {leaderboard_path}")
        print(pd.read_csv(leaderboard_path).to_string(index=False))
        return 0

    task = LeaderboardTask(load_config(args.config))
    if args.command == "validate":
        task.validate()
        _print_json({"valid": True, "profile": task.profile_dataset()})
    elif args.command == "plan":
        task.validate()
        _print_json(task.build_plan())
    else:
        result = task.run()
        _print_json(
            {
                "run_dir": str(result.run_dir),
                "primary_metric": result.primary_metric,
                "best_trial": result.best_trial.to_dict() if result.best_trial else None,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
