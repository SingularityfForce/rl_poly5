from __future__ import annotations
from pathlib import Path
import pandas as pd

from rl_hybrid.data.loaders import load_microstructure, load_cycle_summary
from rl_hybrid.data.alignment import align_datasets, validate_alignment
from rl_hybrid.features.engineering import build_features


def prepare_dataset(micro_paths: list[str], summary_path: str, out_path: str, winner_policy: str = "exclude") -> tuple[pd.DataFrame, dict]:
    micro = load_microstructure(micro_paths)
    summary = load_cycle_summary(summary_path)
    checks = validate_alignment(micro, summary)
    df = align_datasets(micro, summary, winner_policy=winner_policy)
    feat = build_features(df)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(p, index=False)
    return feat, checks
