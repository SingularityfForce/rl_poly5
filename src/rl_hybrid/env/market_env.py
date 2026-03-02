from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_hybrid.features.episodes import Episode
from rl_hybrid.sim.orders import Action, MakerOrder
from rl_hybrid.sim.taker import execute_taker
from rl_hybrid.sim.maker import maker_transition
from rl_hybrid.sim.pnl import mark_to_market, terminal_liquidation


class HybridMarketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode: Episode, feature_cols: list[str], config: dict):
        self.ep = episode
        self.feature_cols = feature_cols
        self.cfg = config
        self.max_inventory = int(self.cfg.get("max_inventory_per_side", 3))
        self.allow_both_sides = bool(self.cfg.get("allow_dual_side_inventory", True))
        self.max_order_age = int(self.cfg.get("max_order_age", 8))

        self.action_space = spaces.Discrete(len(Action))
        obs_dim = len(feature_cols) + 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def _row(self) -> dict:
        return self.ep.data.iloc[self.t].to_dict()

    def _obs(self) -> np.ndarray:
        row = self._row()
        feats = np.array([row.get(c, 0.0) for c in self.feature_cols], dtype=np.float32)
        aux = np.array([
            self.pos_up,
            self.pos_down,
            self.cash,
            self.realized_pnl,
            mark_to_market(self.pos_up, self.pos_down, row),
            len(self.pending),
            self.step_count,
            row.get("time_left_frac", 0.0),
        ], dtype=np.float32)
        return np.concatenate([feats, aux])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos_up = 0
        self.pos_down = 0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.pending: list[MakerOrder] = []
        self.step_count = 0
        return self._obs(), {}

    def _inventory_blocked(self, side: str, qty: int) -> bool:
        cur = self.pos_up if side == "UP" else self.pos_down
        nxt = cur + qty
        if abs(nxt) > self.max_inventory:
            return True
        if not self.allow_both_sides and qty > 0:
            other = self.pos_down if side == "UP" else self.pos_up
            if other > 0:
                return True
        return False

    def step(self, action: int):
        row = self._row()
        prev_mtm = mark_to_market(self.pos_up, self.pos_down, row)
        reward = 0.0

        if action == Action.CANCEL_ALL:
            reward -= 0.0005 * len(self.pending)
            self.pending = []

        elif action in {Action.POST_BID_UP, Action.POST_ASK_UP, Action.POST_BID_DOWN, Action.POST_ASK_DOWN}:
            if len(self.pending) < self.cfg.get("max_active_orders", 4):
                side = "UP" if action in {Action.POST_BID_UP, Action.POST_ASK_UP} else "DOWN"
                is_bid = action in {Action.POST_BID_UP, Action.POST_BID_DOWN}
                px = row.get(f"{side}_{'bid' if is_bid else 'ask'}")
                if px is not None:
                    self.pending.append(MakerOrder(side=side, is_bid=is_bid, px=float(px), qty=1))

        elif action != Action.HOLD:
            filled, side, qty, px, fee = execute_taker(
                action,
                row,
                self.cfg.get("fee_bps", 1.0),
                self.cfg.get("slippage_bps", 2.0),
                self.cfg.get("missing_policy", "reject"),
            )
            if filled and side and not self._inventory_blocked(side, qty):
                self.cash -= qty * px + fee
                if side == "UP":
                    self.pos_up += qty
                else:
                    self.pos_down += qty
                reward -= fee

        new_pending: list[MakerOrder] = []
        for od in self.pending:
            od.age += 1
            if od.age > self.max_order_age:
                reward -= self.cfg.get("stale_penalty", 0.0003)
                continue

            state, px = maker_transition(od, row, self.cfg.get("maker_regime", "base"))
            if state.startswith("fill"):
                qty = 1 if od.is_bid else -1
                if self._inventory_blocked(od.side, qty):
                    continue
                self.cash -= qty * px
                if od.side == "UP":
                    self.pos_up += qty
                else:
                    self.pos_down += qty
                if state == "fill_adverse":
                    reward -= 0.001
            elif state == "stale":
                reward -= self.cfg.get("stale_penalty", 0.0003)
                new_pending.append(od)
            else:
                new_pending.append(od)
        self.pending = new_pending

        self.t += 1
        self.step_count += 1
        terminated = self.t >= len(self.ep.data)
        truncated = False

        if terminated:
            final_val = terminal_liquidation(
                self.cash,
                self.pos_up,
                self.pos_down,
                self.ep.winner,
                self.cfg.get("null_winner_policy", "zero"),
            )
            reward += final_val - self.realized_pnl
            self.realized_pnl = final_val
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            row2 = self._row()
            mtm = mark_to_market(self.pos_up, self.pos_down, row2)
            reward += mtm - prev_mtm
            inv_pen = self.cfg.get("inventory_penalty", 0.0001) * (abs(self.pos_up) + abs(self.pos_down))
            reward -= inv_pen
            obs = self._obs()

        return obs, float(reward), terminated, truncated, {
            "cash": self.cash,
            "pos_up": self.pos_up,
            "pos_down": self.pos_down,
            "pending": len(self.pending),
        }
