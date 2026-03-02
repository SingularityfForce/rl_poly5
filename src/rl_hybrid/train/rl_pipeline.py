from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from rl_hybrid.env.market_env import HybridMarketEnv
from rl_hybrid.features.episodes import Episode
from rl_hybrid.models.dqn import DQNAgent, ReplayBuffer


def train_dqn(episodes: list[Episode], feature_cols: list[str], cfg: dict, outdir: str) -> dict:
    if not episodes:
        raise ValueError("No training episodes available for DQN training")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    env0 = HybridMarketEnv(episodes[0], feature_cols, cfg)
    agent = DQNAgent(
        env0.observation_space.shape[0],
        env0.action_space.n,
        lr=cfg.get("lr", 1e-3),
        gamma=cfg.get("gamma", 0.99),
    )
    rb = ReplayBuffer(cfg.get("buffer_size", 5000))
    eps0, eps1, decay = cfg.get("eps_start", 1.0), cfg.get("eps_end", 0.1), cfg.get("eps_decay", 400)
    total_steps = 0
    best = -1e9
    history: list[float] = []

    for ep_idx in range(cfg.get("train_episodes", 100)):
        ep = episodes[ep_idx % len(episodes)]
        env = HybridMarketEnv(ep, feature_cols, cfg)
        obs, _ = env.reset()
        done = False
        ep_r = 0.0

        while not done:
            eps = eps1 + (eps0 - eps1) * np.exp(-total_steps / max(decay, 1))
            act = agent.act(obs, float(eps))
            nobs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            rb.push(obs, act, r, nobs, done)
            obs = nobs
            ep_r += r
            total_steps += 1

            if len(rb) >= cfg.get("batch_size", 64):
                batch = rb.sample(cfg.get("batch_size", 64))
                agent.update(batch)
            if total_steps % cfg.get("target_update", 200) == 0:
                agent.tgt.load_state_dict(agent.q.state_dict())

        history.append(ep_r)
        if ep_r > best:
            best = ep_r
            torch.save(agent.q.state_dict(), out / "best_dqn.pt")

    return {
        "episodes": len(history),
        "reward_mean": float(np.mean(history)),
        "reward_last": float(history[-1]),
        "reward_best": float(best),
    }
