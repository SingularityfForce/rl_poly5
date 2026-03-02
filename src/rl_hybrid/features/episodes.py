from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class Episode:
    asset: str
    cycle: str
    winner: str | None
    data: pd.DataFrame


def build_episodes(df: pd.DataFrame, action_every_n_ticks: int = 1) -> list[Episode]:
    episodes: list[Episode] = []
    for (asset, cycle), g in df.groupby(["asset", "cycle"], sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        if action_every_n_ticks > 1:
            g = g.iloc[::action_every_n_ticks].reset_index(drop=True)
        if len(g) < 2:
            continue
        episodes.append(Episode(asset=asset, cycle=cycle, winner=g["winner"].iloc[-1], data=g))
    return episodes


def temporal_split(episodes: list[Episode], train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[list[Episode], list[Episode], list[Episode]]:
    eps = sorted(episodes, key=lambda e: (e.asset, e.cycle))
    n = len(eps)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return eps[:n_train], eps[n_train:n_train+n_val], eps[n_train+n_val:]
