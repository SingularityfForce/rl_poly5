from __future__ import annotations
import numpy as np


def summarize_rewards(rewards: list[float]) -> dict[str, float]:
    arr = np.array(rewards, dtype=float)
    eq = np.cumsum(arr)
    dd = np.max(np.maximum.accumulate(eq) - eq) if len(eq) else 0.0
    pos = arr[arr > 0].sum()
    neg = -arr[arr < 0].sum() + 1e-9
    return {
        "reward_total": float(arr.sum()),
        "reward_mean": float(arr.mean()) if len(arr) else 0.0,
        "drawdown": float(dd),
        "profit_factor": float(pos / neg),
        "hit_rate": float((arr > 0).mean()) if len(arr) else 0.0,
    }
