"""Configuration helpers for reproducible GAMERA Ireland workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON configuration file."""
    with Path(path).expanduser().open() as handle:
        return json.load(handle)


def event_window(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return event, start, and end timestamps from an event config."""
    event_time = pd.Timestamp(config["event_time"])
    start = event_time + pd.Timedelta(days=config["start_offset_days"])
    end = event_time + pd.Timedelta(days=config["end_offset_days"])
    return event_time, start, end


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve a path relative to the repository root if it is not absolute."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return Path(project_root).resolve() / raw

