from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    BUY_UP_TAKER = 1
    BUY_DOWN_TAKER = 2
    SELL_UP_TAKER = 3
    SELL_DOWN_TAKER = 4
    POST_BID_UP = 5
    POST_ASK_UP = 6
    POST_BID_DOWN = 7
    POST_ASK_DOWN = 8
    CANCEL_ALL = 9


@dataclass
class MakerOrder:
    side: str  # UP/DOWN
    is_bid: bool
    px: float
    qty: int
    age: int = 0
    stale: bool = False
