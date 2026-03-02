from __future__ import annotations
import numpy as np
import pandas as pd

BASE_COLS = [
    "quoteRate",
    "UP_bid", "UP_ask", "UP_spread", "UP_holeBid", "UP_holeAsk",
    "DOWN_bid", "DOWN_ask", "DOWN_spread", "DOWN_holeBid", "DOWN_holeAsk",
    "UP_bidDepth_1pp", "UP_askDepth_1pp", "UP_imb_1pp", "DOWN_bidDepth_1pp", "DOWN_askDepth_1pp", "DOWN_imb_1pp",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_features(df: pd.DataFrame, rolling_windows: list[int] | None = None) -> pd.DataFrame:
    rolling_windows = rolling_windows or [3, 8, 16]
    x = df.copy()
    x["implied_up"] = _safe_div(x["UP_bid"].fillna(0) + (1 - x["DOWN_ask"].fillna(1)), pd.Series(2, index=x.index)).fillna(0.5)
    x["thinness"] = ((x["UP_bidDepth_1pp"].fillna(0) + x["DOWN_bidDepth_1pp"].fillna(0)) < 100).astype(float)
    x["spread_asym"] = x["UP_spread"].fillna(0) - x["DOWN_spread"].fillna(0)
    x["depth_asym"] = x["UP_bidDepth_1pp"].fillna(0) - x["DOWN_bidDepth_1pp"].fillna(0)
    x["time_in_cycle"] = x.groupby(["asset", "cycle"]).cumcount()
    max_t = x.groupby(["asset", "cycle"])["time_in_cycle"].transform("max").replace(0, 1)
    x["time_left_frac"] = 1 - x["time_in_cycle"] / max_t
    for c in ["UP_bid", "UP_ask", "DOWN_bid", "DOWN_ask", "quoteRate"]:
        x[f"d_{c}"] = x.groupby(["asset", "cycle"])[c].diff().fillna(0)
    for w in rolling_windows:
        for c in ["implied_up", "quoteRate", "UP_spread", "DOWN_spread"]:
            grp = x.groupby(["asset", "cycle"])[c]
            x[f"{c}_mean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
            x[f"{c}_std_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
    for c in ["UP_bid", "UP_ask", "DOWN_bid", "DOWN_ask"]:
        x[f"missing_{c}"] = x[c].isna().astype(float)
    x = x.fillna(0)
    return x
