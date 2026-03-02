from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
