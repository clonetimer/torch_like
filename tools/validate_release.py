"""Run every collected test exactly once in bounded, interpreter-isolated groups.

Usage: python tools/validate_release.py --all
       python tools/validate_release.py --list
       python tools/validate_release.py --group core

Writes durable logs, JUnit XML, per-rank attestations and an aggregate manifest.
This tool makes no repository-wide lint, GPU, performance or coverage claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
FILE_GROUPS = {
    "models": [
        "audio_api",
        "detection_api",
        "egnn",
        "forecasting_api",
        "generative_api",
        "mamba2",
        "multimodal_api",
        "recommendation_api",
        "reinforcement_api",
        "tabular_api",
        "task_interfaces",
        "vision_api",
    ],
    "core": [
        "api_stability",
        "capabilities",
        "catalog_evidence",
        "contracts_runtime",
        "core",
        "engineering_api",
        "lazy_imports",
        "model_catalog_audit",
        "registration_scope",
        "registry",
    ],
    "automl_support": [
        "automl_examples",
        "automl_protocol",
        "automl_run_validation",
        "backend_runners",
        "candidate_pool",
        "run_events",
    ],
    "training_runner": [
        "runner_api",
        "task_data_api",
        "training_adapters_evaluation",
        "training_events",
        "training_runtime",
    ],
    "checkpoint_compatibility": ["checkpoint_compatibility"],
}


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _digest() -> str:
    files = [ROOT / "pyproject.toml"]
    for name in ("torch_like", "runner", "tests", "tools", "governance"):
        files.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and p.suffix in {".py", ".json"})
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _collect() -> list[str]:
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    (OUTPUT / "release_collection.log").write_text(collected.stdout, encoding="utf-8")
    if collected.returncode:
        raise RuntimeError("test collection failed; see validation/release_collection.log")
    return [line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]


def _groups(nodes: list[str]) -> dict[str, list[str]]:
    remaining = set(nodes)
    groups = {}
    for key, files in FILE_GROUPS.items():
        paths = {f"tests/test_{name}.py" for name in files}
        group = [node for node in nodes if node.split("::")[0] in paths]
        groups[key] = group
        remaining.difference_update(group)
    for cls in ("AutoMLDataTests", "AutoMLBackendTests"):
        group = [node for node in nodes if node.startswith(f"tests/test_automl.py::{cls}::")]
        groups[cls] = group
        remaining.difference_update(group)
    tasks = [node for node in nodes if node.startswith("tests/test_automl.py::AutoMLTaskTests::")]
    for index, node in enumerate(tasks):
        groups[f"automl_task_{index:02d}"] = [node]
        remaining.remove(node)
    ddp = [node for node in nodes if node.startswith("tests/test_ddp_runtime.py::test_real_two_rank_gloo")]
    for offset in range(0, len(ddp), 3):
        group = ddp[offset : offset + 3]
        groups[f"ddp_{offset // 3:02d}"] = group
        remaining.difference_update(group)
    ddp_units = [node for node in nodes if node.startswith("tests/test_ddp_runtime.py::") and node in remaining]
    groups["ddp_units"] = ddp_units
    remaining.difference_update(ddp_units)
    if remaining:
        groups["other"] = sorted(remaining)
    planned = [node for group in groups.values() for node in group]
    if len(planned) != len(set(planned)) or set(planned) != set(nodes):
        raise RuntimeError("test grouping overlaps or omits collected tests")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    selection.add_argument("--group", action="append")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    OUTPUT.mkdir(exist_ok=True)
    nodes = _collect()
    groups = _groups(nodes)
    if args.list:
        print(json.dumps({key: len(value) for key, value in groups.items()}, indent=2))
        return 0
    requested = list(groups) if args.all else args.group
    if any(key not in groups for key in requested):
        parser.error("unknown group; use --list")
    digest = _digest()
    manifest_path = OUTPUT / "release_validation.json"
    manifest = {"source_code_digest": digest, "collected": len(nodes), "groups": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("source_code_digest") == digest and not args.all:
            manifest = previous
    workspace = Path(tempfile.mkdtemp(prefix="torch_like_release_"))
    try:
        for key in requested:
            if _digest() != digest:
                raise RuntimeError("source code changed during validation; restart on a frozen tree")
            manifest["running_group"] = key
            _write(manifest_path, manifest)
            xml = OUTPUT / f"release_{key}.xml"
            log = OUTPUT / f"release_{key}.log"
            xml.unlink(missing_ok=True)
            base = workspace / key
            command = [sys.executable, "-m", "pytest", *groups[key], "-q", f"--junitxml={xml}", f"--basetemp={base}"]
            started = time.monotonic()
            timed_out = False
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=ENV,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
                    code = process.wait()
            cases = list(ET.parse(xml).iter("testcase")) if xml.exists() else []
            failures = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
            skipped = sum(case.find("skipped") is not None for case in cases)
            passed = len(cases) - failures - skipped
            success = not timed_out and code == 0 and len(cases) == len(groups[key]) and skipped == 0
            result = {
                "nodeids": groups[key],
                "returncode": code,
                "timeout": timed_out,
                "seconds": round(time.monotonic() - started, 3),
                "executed": len(cases),
                "passed": passed,
                "failed": failures,
                "skipped": skipped,
                "success": success,
                "log": log.name,
                "junit": xml.name,
            }
            manifest["groups"][key] = result
            attestations = {
                path.relative_to(base).as_posix(): json.loads(path.read_text())
                for path in base.rglob("rank*.json")
                if not path.is_symlink()
            }
            if attestations:
                _write(OUTPUT / f"release_{key}_rank_results.json", attestations)
            # A timed-out launcher must not leave descendants behind on POSIX.
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            shutil.rmtree(base, ignore_errors=True)
            _write(manifest_path, manifest)
            print(f"{key}: passed={passed}, failed={failures}, skipped={skipped}, success={success}", flush=True)
        manifest.pop("running_group", None)
        executed = {node for result in manifest["groups"].values() for node in result["nodeids"]}
        manifest["unique_selected"] = len(executed)
        manifest["passed"] = sum(result["passed"] for result in manifest["groups"].values())
        manifest["failed"] = sum(result["failed"] for result in manifest["groups"].values())
        manifest["skipped"] = sum(result["skipped"] for result in manifest["groups"].values())
        manifest["complete"] = executed == set(nodes) and all(r["success"] for r in manifest["groups"].values())
        manifest["missing"] = sorted(set(nodes) - executed)
        _write(manifest_path, manifest)
        return 0 if all(manifest["groups"][key]["success"] for key in requested) else 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
