from __future__ import annotations

import gzip
import json
import logging
import zipfile
from pathlib import Path
from io import TextIOWrapper
from typing import Iterable
import pandas as pd
from pydantic import ValidationError

from .schema import MicroRecord, CycleSummary, flatten_micro

logger = logging.getLogger(__name__)


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            if not members:
                logger.warning("zip archive has no files: %s", path)
                return
            for member in members:
                is_jsonl_like = member.endswith(".jsonl") or member.endswith(".jsonl.gz")
                if not is_jsonl_like:
                    logger.debug("skipping non-jsonl member %s in %s", member, path)
                    continue
                with zf.open(member, "r") as raw:
                    stream = gzip.open(raw, "rt", encoding="utf-8") if member.endswith(".gz") else TextIOWrapper(raw, encoding="utf-8")
                    with stream as f:
                        for ln, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError as e:
                                logger.warning("invalid json line %d in %s::%s: %s", ln, path, member, e)
        return

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
