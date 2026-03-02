from __future__ import annotations
import math
import random
from .orders import MakerOrder


REGIME_MULT = {
    "optimistic": {"fill": 1.2, "adv": 0.8, "stale": 0.8},
    "base": {"fill": 1.0, "adv": 1.0, "stale": 1.0},
    "pessimistic": {"fill": 0.7, "adv": 1.4, "stale": 1.3},
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(min(x, 10), -10)))


def maker_transition(order: MakerOrder, row: dict, regime: str = "base") -> tuple[str, float]:
    m = REGIME_MULT[regime]
    spread = float(row.get(f"{order.side}_spread", 0.02) or 0.02)
    imb = float(row.get(f"{order.side}_imb_1pp", 0.0) or 0.0)
    hole = float(row.get(f"{order.side}_{'holeBid' if order.is_bid else 'holeAsk'}", 0.0) or 0.0)
    quote_rate = float(row.get("quoteRate", 0.0) or 0.0)
    tleft = float(row.get("time_left_frac", 0.5) or 0.5)

    tox = 2.0 * quote_rate + 1.5 * abs(hole) + (1 - tleft)
    fill_p = _sigmoid(-2.5 * spread + (0.5 if abs(imb) < 0.2 else -0.5) - 0.8 * tox - 0.05 * order.age) * m["fill"]
    stale_p = _sigmoid(0.8 * tox + 0.07 * order.age) * 0.5 * m["stale"]
    adv_p = _sigmoid(0.8 * tox + (-imb if order.is_bid else imb)) * m["adv"]

    fill_p = min(max(fill_p, 0.01), 0.9)
    stale_p = min(max(stale_p, 0.01), 0.95)
    adv_p = min(max(adv_p, 0.02), 0.98)

    r = random.random()
    if r < fill_p:
        adverse = random.random() < adv_p
        slip = 0.002 if adverse else -0.001
        px = max(0.001, min(0.999, order.px + slip if order.is_bid else order.px - slip))
        return ("fill_adverse" if adverse else "fill_favorable"), px
    if r < fill_p + stale_p:
        return "stale", order.px
    return "pending", order.px
