from __future__ import annotations
import math


def execute_taker(action: int, row: dict, fee_bps: float, slippage_bps: float, missing_policy: str = "reject") -> tuple[bool, str | None, int, float, float]:
    # filled, side, signed_qty, exec_price, fee
    def _price(side: str, buy: bool) -> float | None:
        k = f"{side}_{'ask' if buy else 'bid'}"
        return row.get(k)

    mapping = {
        1: ("UP", True, 1),
        2: ("DOWN", True, 1),
        3: ("UP", False, -1),
        4: ("DOWN", False, -1),
    }
    if action not in mapping:
        return False, None, 0, 0.0, 0.0
    side, buy, qty = mapping[action]
    px = _price(side, buy)
    if px is None or (isinstance(px, float) and math.isnan(px)):
        if missing_policy == "penalized":
            px = 0.99 if buy else 0.01
        else:
            return False, None, 0, 0.0, 0.0
    px = float(px) * (1 + slippage_bps / 1e4 if buy else 1 - slippage_bps / 1e4)
    fee = abs(px * qty) * fee_bps / 1e4
    return True, side, qty, px, fee
