from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_hybrid.env.market_env import HybridMarketEnv
from rl_hybrid.models.dqn import QNet
from rl_hybrid.eval.metrics import summarize_rewards


def run_backtest(episodes, feature_cols, cfg, model_path: str, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rewards = []
    for regime in ["optimistic", "base", "pessimistic"]:
        reg_rewards = []
        env0 = HybridMarketEnv(episodes[0], feature_cols, {**cfg, "maker_regime": regime})
        model = QNet(env0.observation_space.shape[0], env0.action_space.n)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        for ep in episodes:
            env = HybridMarketEnv(ep, feature_cols, {**cfg, "maker_regime": regime})
            obs, _ = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                with torch.no_grad():
                    a = int(model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).argmax(1).item())
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_r += r
            reg_rewards.append(ep_r)
        rewards.append((regime, reg_rewards))
    report = {k: summarize_rewards(v) for k, v in rewards}
    with (out / "backtest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    plt.figure(figsize=(7, 4))
    for k, v in rewards:
        plt.plot(np.cumsum(v), label=k)
    plt.legend(); plt.title("Equity by maker regime")
    plt.savefig(out / "equity_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    return report
