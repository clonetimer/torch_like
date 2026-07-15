"""YAML and JSON configuration loading for leaderboard tasks."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from torch_like.automl.specs import LeaderboardConfig


def load_config(path: str | Path) -> LeaderboardConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Leaderboard configuration does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        raise ValueError("Leaderboard configuration must use .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError("Leaderboard configuration root must be a mapping")
    return LeaderboardConfig.from_dict(payload)
