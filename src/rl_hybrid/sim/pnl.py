from __future__ import annotations


def mark_to_market(pos_up: int, pos_down: int, row: dict) -> float:
    up_mid = ((row.get("UP_bid") or 0.5) + (row.get("UP_ask") or 0.5)) / 2
    down_mid = ((row.get("DOWN_bid") or 0.5) + (row.get("DOWN_ask") or 0.5)) / 2
    return pos_up * up_mid + pos_down * down_mid


def terminal_liquidation(cash: float, pos_up: int, pos_down: int, winner: str | None, null_policy: str = "zero") -> float:
    if winner is None:
        if null_policy == "zero":
            return cash
    payoff_up = 1.0 if winner == "UP" else 0.0
    payoff_down = 1.0 if winner == "DOWN" else 0.0
    return cash + pos_up * payoff_up + pos_down * payoff_down
