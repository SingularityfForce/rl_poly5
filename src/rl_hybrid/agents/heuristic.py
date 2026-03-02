from __future__ import annotations


def heuristic_action(row: dict, inv_up: int, inv_down: int) -> int:
    quote = float(row.get("quoteRate", 0.0) or 0.0)
    thin = float(row.get("thinness", 0.0) or 0.0)
    tleft = float(row.get("time_left_frac", 0.5) or 0.5)
    implied = float(row.get("implied_up", 0.5) or 0.5)
    toxic = quote + thin + (1 - tleft)
    if toxic > 1.4:
        return 9  # cancel
    if implied > 0.58 and inv_up <= 0:
        return 1
    if implied < 0.42 and inv_down <= 0:
        return 2
    if toxic < 0.7:
        return 5 if implied >= 0.5 else 7
    return 0
