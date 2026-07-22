"""Validate completed AutoML leaderboard run artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from torch_like.automl import LeaderboardConfig, LeaderboardTrustLevel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs"
REQUIRED_ARTIFACTS = (
    "config.json",
    "dataset_profile.json",
    "data_split.json",
    "environment.json",
    "plan.json",
    "protocol_audit.json",
    "evaluation_trust_audit.json",
    "trials.json",
    "leaderboard.csv",
    "sealed_test_metrics.json",
    "final_report.md",
)


@dataclass(frozen=True)
class RunValidationResult:
    path: Path
    errors: tuple[str, ...]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_run_dirs(root: Path) -> list[Path]:
    if (root / "config.json").is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "config.json").is_file())


def _validate_required_files(run_dir: Path) -> list[str]:
    return [f"missing required artifact: {name}" for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]


def _validate_validated_run(
    config: LeaderboardConfig,
    plan: dict[str, Any],
    trust_audit: dict[str, Any],
    leaderboard: pd.DataFrame,
    sealed: dict[str, Any],
) -> list[str]:
    errors = []
    seeds = config.search.evaluation_seeds or (config.seed,)
    if len(seeds) < 3:
        errors.append("validated run requires at least three evaluation seeds")
    if plan.get("trust_level") != LeaderboardTrustLevel.VALIDATED.value:
        errors.append("validated run plan must record trust_level='validated'")
    if trust_audit.get("level") != LeaderboardTrustLevel.VALIDATED.value:
        errors.append("validated run audit must record level='validated'")
    if trust_audit.get("details", {}).get("minimum_evaluation_seeds", 0) < 3:
        errors.append("validated run audit must record minimum_evaluation_seeds >= 3")
    completed = leaderboard[leaderboard["status"] == "completed"] if "status" in leaderboard else leaderboard.iloc[0:0]
    if completed.empty:
        errors.append("validated run requires at least one completed ranked candidate")
    elif not (completed["completed_seed_count"].astype(int) == completed["seed_count"].astype(int)).all():
        errors.append("validated completed candidates must have completed_seed_count == seed_count")
    if sealed.get("status") != "completed":
        errors.append("validated run requires completed sealed_test_metrics.json")
    return errors


def validate_run(run_dir: str | Path) -> RunValidationResult:
    path = Path(run_dir)
    errors = _validate_required_files(path)
    if errors:
        return RunValidationResult(path=path, errors=tuple(errors))
    try:
        config = LeaderboardConfig.from_dict(_load_json(path / "config.json"))
        plan = _load_json(path / "plan.json")
        protocol_audit = _load_json(path / "protocol_audit.json")
        trust_audit = _load_json(path / "evaluation_trust_audit.json")
        sealed = _load_json(path / "sealed_test_metrics.json")
        leaderboard = pd.read_csv(path / "leaderboard.csv")
        trials = _load_json(path / "trials.json")
    except Exception as exc:  # noqa: BLE001 - preserve path context for batch reporting
        return RunValidationResult(path=path, errors=(f"failed to read run artifacts: {exc}",))

    if protocol_audit.get("status") != "passed":
        errors.append("protocol_audit.json must have status='passed'")
    if trust_audit.get("status") != "passed":
        errors.append("evaluation_trust_audit.json must have status='passed'")
    if plan.get("task_id") != config.task_id:
        errors.append("plan task_id must match config task_id")
    if plan.get("trust_level") != config.trust.level.value:
        errors.append("plan trust_level must match config trust.level")
    if trust_audit.get("level") != config.trust.level.value:
        errors.append("evaluation_trust_audit level must match config trust.level")
    if plan.get("trial_count") != len(trials):
        errors.append("plan trial_count must equal trials.json length")
    if leaderboard.empty:
        errors.append("leaderboard.csv must contain at least one candidate row")
    else:
        required_columns = {"trial_id", "status", "seed_count", "completed_seed_count", "evaluation_seeds"}
        missing_columns = required_columns - set(leaderboard.columns)
        if missing_columns:
            errors.append(f"leaderboard.csv missing required columns: {sorted(missing_columns)}")
        if any(column.startswith("test_") for column in leaderboard.columns):
            errors.append("leaderboard.csv must not expose test_* metrics before sealed-test selection")
    if sealed.get("status") == "completed":
        if not sealed.get("selected_trial_id"):
            errors.append("completed sealed_test_metrics.json must record selected_trial_id")
        if not sealed.get("test_metrics"):
            errors.append("completed sealed_test_metrics.json must record test_metrics")
    if config.trust.level == LeaderboardTrustLevel.VALIDATED:
        errors.extend(_validate_validated_run(config, plan, trust_audit, leaderboard, sealed))
    return RunValidationResult(path=path, errors=tuple(errors))


def validate_runs(paths: list[str | Path], *, skip_legacy: bool = False) -> list[RunValidationResult]:
    results = []
    for path in paths:
        root = Path(path)
        run_dirs = _candidate_run_dirs(root)
        if not run_dirs:
            results.append(RunValidationResult(path=root, errors=("no AutoML run directories found",)))
            continue
        if skip_legacy:
            run_dirs = [
                run_dir
                for run_dir in run_dirs
                if (run_dir / "protocol_audit.json").is_file() and (run_dir / "evaluation_trust_audit.json").is_file()
            ]
            if not run_dirs:
                results.append(RunValidationResult(path=root, errors=("no current-contract run directories found",)))
                continue
        results.extend(validate_run(run_dir) for run_dir in run_dirs)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_RUNS_ROOT])
    parser.add_argument("--allow-empty", action="store_true", help="exit successfully when no run directories exist")
    parser.add_argument("--skip-legacy", action="store_true", help="skip pre-trust-audit run directories")
    args = parser.parse_args()
    results = validate_runs(args.paths, skip_legacy=args.skip_legacy)
    if args.allow_empty:
        results = [
            result
            for result in results
            if result.errors
            not in {
                ("no AutoML run directories found",),
                ("no current-contract run directories found",),
            }
        ]
    errors = [(result.path, error) for result in results for error in result.errors]
    if errors:
        for path, error in errors:
            print(f"{path}: {error}")
        return 1
    print(f"AutoML run validation passed: {len(results)} run directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
