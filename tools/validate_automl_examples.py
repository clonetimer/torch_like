"""Validate AutoML example configuration files without reading external datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from torch_like.automl import LeaderboardConfig, LeaderboardTrustLevel, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"


@dataclass(frozen=True)
class ExampleValidationResult:
    path: Path
    config: LeaderboardConfig | None
    errors: tuple[str, ...]


def _seed_count(config: LeaderboardConfig) -> int:
    return len(config.search.evaluation_seeds or (config.seed,))


def _validate_config(path: Path, config: LeaderboardConfig) -> list[str]:
    errors = []
    name = path.name.lower()
    task_id = config.task_id.lower()
    if config.trust.level == LeaderboardTrustLevel.VALIDATED:
        if _seed_count(config) < 3:
            errors.append("validated examples must declare at least three evaluation seeds")
        if not any(marker in name or marker in task_id for marker in ("validated", "benchmark")):
            errors.append("validated examples must include 'validated' or 'benchmark' in the file name or task_id")
    if ("validated" in name or "validated" in task_id) and config.trust.level != LeaderboardTrustLevel.VALIDATED:
        errors.append("examples named validated must set trust.level='validated'")
    if ("smoke" in name or "smoke" in task_id) and config.trust.level != LeaderboardTrustLevel.DEVELOPMENT:
        errors.append("smoke examples must keep trust.level='development'")
    if "benchmark" in name and config.trust.level == LeaderboardTrustLevel.DEVELOPMENT and _seed_count(config) == 1:
        errors.append("benchmark examples must use multiple seeds or avoid benchmark naming")
    if config.search.max_trials < len(config.search.algorithms or ()):
        errors.append("max_trials should cover every explicitly listed algorithm at least once")
    if config.artifacts.output_dir in {"", "."}:
        errors.append("artifacts.output_dir must be an explicit run directory")
    return errors


def validate_examples(examples_dir: str | Path = EXAMPLES_DIR) -> list[ExampleValidationResult]:
    root = Path(examples_dir)
    results = []
    task_ids: dict[str, Path] = {}
    for path in sorted(root.glob("*.yaml")):
        try:
            config = load_config(path)
        except Exception as exc:  # noqa: BLE001 - report all examples instead of stopping at the first one
            results.append(ExampleValidationResult(path=path, config=None, errors=(str(exc),)))
            continue
        errors = _validate_config(path, config)
        previous = task_ids.setdefault(config.task_id, path)
        if previous != path:
            errors.append(f"duplicate task_id {config.task_id!r}; first seen in {previous}")
        results.append(ExampleValidationResult(path=path, config=config, errors=tuple(errors)))
    if not results:
        results.append(ExampleValidationResult(path=root, config=None, errors=("no AutoML YAML examples found",)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()
    results = validate_examples(args.examples_dir)
    errors = [(result.path, error) for result in results for error in result.errors]
    if errors:
        for path, error in errors:
            print(f"{path}: {error}")
        return 1
    print(f"AutoML example validation passed: {len(results)} YAML files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
