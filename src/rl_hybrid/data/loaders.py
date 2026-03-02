from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable
import pandas as pd
from pydantic import ValidationError

from .schema import MicroRecord, CycleSummary, flatten_micro

logger = logging.getLogger(__name__)


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("invalid json line %d in %s: %s", ln, path, e)


def load_microstructure(paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for rec in _iter_jsonl(path):
            try:
                m = MicroRecord.model_validate(rec)
                rows.append(flatten_micro(m.model_dump()))
            except ValidationError:
                continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["asset", "cycle", "ts"]).drop_duplicates(["asset", "cycle", "ts"], keep="last")
    return df


def load_cycle_summary(path: str | Path) -> pd.DataFrame:
    rows = []
    for rec in _iter_jsonl(path):
        try:
            c = CycleSummary.model_validate(rec)
        except ValidationError:
            continue
        d = c.model_dump()
        if d.get("type") != "cycle":
            continue
        final = d.pop("final") or {}
        for k, v in final.items():
            d[f"final_{k}"] = v
        rows.append(d)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["asset", "cycle"]).drop_duplicates(["asset", "cycle"], keep="last")
    return df
