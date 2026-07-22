import json
from pathlib import Path

import pandas as pd

from tools.validate_automl_runs import validate_run, validate_runs


def assert_no_errors(result_or_results) -> None:
    results = result_or_results if isinstance(result_or_results, list) else [result_or_results]
    errors = [(result.path, error) for result in results for error in result.errors]
    assert errors == []


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run(root: Path, *, trust_level: str = "development", seeds: tuple[int, ...] = (42,)) -> Path:
    run_dir = root / f"{trust_level}-run"
    run_dir.mkdir()
    config = {
        "task_id": f"{trust_level}-run",
        "dataset": {"path": str(root / "data.csv"), "label_columns": ["label"]},
        "task": {"type": "tabular_classification"},
        "search": {"evaluation_seeds": list(seeds)} if seeds != (42,) else {},
        "trust": {"level": trust_level},
        "artifacts": {"output_dir": str(root)},
    }
    plan = {
        "task_id": config["task_id"],
        "trust_level": trust_level,
        "trial_count": len(seeds),
        "evaluation_seeds": list(seeds),
    }
    trust_audit = {
        "status": "passed",
        "level": trust_level,
        "checks": ["minimum_seed_count_met", "every_candidate_gets_every_seed"],
        "warnings": [] if trust_level == "validated" else ["single_seed_development_run"],
        "details": {
            "evaluation_seeds": list(seeds),
            "minimum_evaluation_seeds": 3 if trust_level == "validated" else 1,
        },
    }
    sealed = {
        "status": "completed",
        "selected_trial_id": "trial-0000",
        "selected_pipeline": "sklearn:standard+LogisticRegression",
        "test_metrics": {"f1_macro": 1.0},
    }
    leaderboard = pd.DataFrame(
        [
            {
                "trial_id": "trial-0000",
                "status": "completed",
                "seed_count": len(seeds),
                "completed_seed_count": len(seeds),
                "evaluation_seeds": ",".join(str(seed) for seed in seeds),
                "val_f1_macro": 1.0,
            }
        ]
    )
    trials = [
        {
            "trial_id": f"trial-0000-seed-{seed:010d}" if len(seeds) > 1 else "trial-0000",
            "status": "completed",
            "seed": seed,
        }
        for seed in seeds
    ]
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "dataset_profile.json", {"fingerprint": "unit-test"})
    _write_json(run_dir / "data_split.json", {"train": [0], "val": [1], "test": [2]})
    _write_json(run_dir / "environment.json", {"git_commit": "unit-test"})
    _write_json(run_dir / "plan.json", plan)
    _write_json(run_dir / "protocol_audit.json", {"status": "passed", "checks": ["sealed_test_isolated"]})
    _write_json(run_dir / "evaluation_trust_audit.json", trust_audit)
    _write_json(run_dir / "sealed_test_metrics.json", sealed)
    _write_json(run_dir / "trials.json", trials)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    (run_dir / "final_report.md").write_text("# Unit test\n", encoding="utf-8")
    return run_dir


def test_validate_development_run_artifacts(tmp_path):
    run_dir = _write_run(tmp_path)

    assert_no_errors(validate_run(run_dir))


def test_validate_validated_run_artifacts(tmp_path):
    run_dir = _write_run(tmp_path, trust_level="validated", seeds=(11, 29, 47))

    assert_no_errors(validate_run(run_dir))
    assert_no_errors(validate_runs([tmp_path]))


def test_validate_validated_run_rejects_incomplete_seed_coverage(tmp_path):
    run_dir = _write_run(tmp_path, trust_level="validated", seeds=(11, 29, 47))
    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    leaderboard.loc[0, "completed_seed_count"] = 2
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    result = validate_run(run_dir)

    assert "validated completed candidates must have completed_seed_count == seed_count" in result.errors


def test_validate_runs_can_skip_legacy_runs(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_json(legacy / "config.json", {"task_id": "legacy"})

    result = validate_runs([tmp_path], skip_legacy=True)

    assert result[0].errors == ("no current-contract run directories found",)
