from __future__ import annotations

import pandas as pd


def align_datasets(micro: pd.DataFrame, summary: pd.DataFrame, winner_policy: str = "exclude") -> pd.DataFrame:
    """Align microstructure ticks with cycle summary by (asset, cycle)."""
    if micro.empty:
        return micro.copy()

    s_cols = ["asset", "cycle", "winner", "firstTs", "lastTs", "n", "thinPct", "quoteRateMean"]
    use_cols = [c for c in s_cols if c in summary.columns]
    merged = micro.merge(summary[use_cols], on=["asset", "cycle"], how="left")
    merged = merged.sort_values(["asset", "cycle", "ts"]).reset_index(drop=True)

    if winner_policy == "exclude":
        merged = merged[merged["winner"].notna()].reset_index(drop=True)
    elif winner_policy == "truncate":
        merged.loc[merged["winner"].isna(), "type"] = "end"
    elif winner_policy not in {"keep"}:
        raise ValueError(f"winner_policy unsupported: {winner_policy}")

    return merged


def validate_alignment(micro: pd.DataFrame, summary: pd.DataFrame, expected_assets: list[str] | None = None) -> dict[str, int]:
    """Return quality checks required for robust ingestion."""
    if micro.empty:
        return {
            "cycles_without_summary": 0,
            "summary_without_ticks": len(summary),
            "timestamp_disorder": 0,
            "duplicate_ticks": 0,
            "unexpected_assets": 0,
            "gap_count_large": 0,
        }

    micro_keys = set(zip(micro.asset, micro.cycle))
    sum_keys = set(zip(summary.asset, summary.cycle)) if not summary.empty else set()

    ts_diff = micro.groupby(["asset", "cycle"]).ts.diff()
    gaps = int((ts_diff > ts_diff.quantile(0.99)).fillna(False).sum()) if len(ts_diff.dropna()) else 0
    unexpected_assets = 0
    if expected_assets:
        unexpected_assets = int((~micro["asset"].isin(expected_assets)).sum())

    duplicate_ticks = int(micro.duplicated(["asset", "cycle", "ts"]).sum())

    return {
        "cycles_without_summary": len(micro_keys - sum_keys),
        "summary_without_ticks": len(sum_keys - micro_keys),
        "timestamp_disorder": int((ts_diff < 0).sum()),
        "duplicate_ticks": duplicate_ticks,
        "unexpected_assets": unexpected_assets,
        "gap_count_large": gaps,
    }
