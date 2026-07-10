"""Utilities for summarizing runner result JSON files."""

from __future__ import annotations

from pathlib import Path
import json


def load_result_jsons(results_dir: str | Path):
    results_dir = Path(results_dir)
    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payloads.append({"path": str(path), "payload": payload})
    return payloads


def summarize_results(results_dir: str | Path):
    summaries = []
    for item in load_result_jsons(results_dir):
        payload = item["payload"]
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        results = payload.get("results")
        summary = {
            "path": item["path"],
            "task": payload.get("extra", {}).get("task"),
            "history_len": len(payload.get("history", [])),
            "config_keys": sorted(config.keys()) if isinstance(config, dict) else [],
            "has_metrics": bool(metrics),
            "has_results": results is not None,
        }
        if isinstance(metrics, dict) and "test" in metrics:
            summary["test_metrics"] = metrics["test"]
        summaries.append(summary)
    return summaries
