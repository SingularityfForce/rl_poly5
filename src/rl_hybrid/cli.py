from __future__ import annotations

import json
import pandas as pd
import typer

from rl_hybrid.utils.config import load_config
from rl_hybrid.utils.logging_utils import setup_logging
from rl_hybrid.utils.seeding import seed_everything
from rl_hybrid.train.data_pipeline import prepare_dataset
from rl_hybrid.features.episodes import build_episodes, temporal_split
from rl_hybrid.train.supervised_pipeline import train_supervised
from rl_hybrid.train.rl_pipeline import train_dqn
from rl_hybrid.eval.backtest import run_backtest

app = typer.Typer()


def _last_tick_df(episodes):
    if not episodes:
        return pd.DataFrame()
    return pd.concat([e.data.iloc[[-1]] for e in episodes], ignore_index=True)


@app.command()
def prep_data(config: str = "configs/base.yaml"):
    cfg = load_config(config)
    setup_logging(cfg.get("log_level", "INFO"))
    df, checks = prepare_dataset(
        cfg["micro_paths"],
        cfg["summary_path"],
        cfg["dataset_path"],
        cfg.get("winner_policy", "exclude"),
    )
    typer.echo(f"rows={len(df)} checks={checks}")


@app.command()
def train_sup(config: str = "configs/train_supervised.yaml"):
    cfg = load_config(config)
    seed_everything(cfg.get("seed", 7))
    df = pd.read_parquet(cfg["dataset_path"])
    episodes = build_episodes(df, action_every_n_ticks=cfg.get("action_every_n_ticks", 1))
    tr, va, te = temporal_split(episodes, cfg.get("train_frac", 0.7), cfg.get("val_frac", 0.15))

    if not tr or not va or not te:
        raise typer.BadParameter("Temporal split produced empty train/val/test sets. Increase data or adjust fractions.")

    metrics = train_supervised(
        _last_tick_df(tr),
        _last_tick_df(va),
        _last_tick_df(te),
        cfg["feature_cols"],
        cfg["outdir"],
    )
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def train_rl(config: str = "configs/train_rl.yaml"):
    cfg = load_config(config)
    seed_everything(cfg.get("seed", 7))
    df = pd.read_parquet(cfg["dataset_path"])
    episodes = build_episodes(df, action_every_n_ticks=cfg.get("action_every_n_ticks", 1))
    tr, _, _ = temporal_split(episodes, cfg.get("train_frac", 0.7), cfg.get("val_frac", 0.15))
    metrics = train_dqn(tr, cfg["feature_cols"], cfg, cfg["outdir"])
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def backtest(config: str = "configs/backtest.yaml"):
    cfg = load_config(config)
    df = pd.read_parquet(cfg["dataset_path"])
    episodes = build_episodes(df, action_every_n_ticks=cfg.get("action_every_n_ticks", 1))
    _, _, te = temporal_split(episodes, cfg.get("train_frac", 0.7), cfg.get("val_frac", 0.15))
    if not te:
        raise typer.BadParameter("No test episodes available for backtest")
    report = run_backtest(te, cfg["feature_cols"], cfg, cfg["model_path"], cfg["outdir"])
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
