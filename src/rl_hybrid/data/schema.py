from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field


class SideQuote(BaseModel):
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    holeBid: float | None = None
    holeAsk: float | None = None
    bidDepth_1pp: float | None = None
    askDepth_1pp: float | None = None
    imb_1pp: float | None = None
    bidDepth_2pp: float | None = None
    askDepth_2pp: float | None = None
    imb_2pp: float | None = None
    bidDepth_5pp: float | None = None
    askDepth_5pp: float | None = None
    imb_5pp: float | None = None


class MicroRecord(BaseModel):
    type: Literal["tick", "rollover", "end"]
    ts: int
    asset: str
    cycle: str
    quoteRate: float | None = None
    UP: SideQuote = Field(default_factory=SideQuote)
    DOWN: SideQuote = Field(default_factory=SideQuote)


class CycleFinal(BaseModel):
    UP_bid: float | None = None
    UP_ask: float | None = None
    DOWN_bid: float | None = None
    DOWN_ask: float | None = None
    impliedUp: float | None = None


class CycleSummary(BaseModel):
    type: Literal["start", "cycle", "end"]
    asset: str | None = None
    cycle: str | None = None
    firstTs: int | None = None
    lastTs: int | None = None
    n: int | None = None
    thinPct: float | None = None
    quoteRateMean: float | None = None
    spreadUpMean: float | None = None
    spreadDownMean: float | None = None
    impliedUpMean: float | None = None
    impliedUpMin: float | None = None
    impliedUpMax: float | None = None
    final: CycleFinal | None = None
    winner: Literal["UP", "DOWN"] | None = None


def flatten_micro(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: rec.get(k) for k in ["type", "ts", "asset", "cycle", "quoteRate"]}
    for side in ("UP", "DOWN"):
        d = rec.get(side, {}) or {}
        for k, v in d.items():
            out[f"{side}_{k}"] = v
    return out
