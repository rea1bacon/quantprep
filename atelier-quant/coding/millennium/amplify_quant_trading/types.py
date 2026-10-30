"""Shared value types used across the package.

These mirror (in spirit) the kind of small, encapsulated data objects the real
AmplifyQuantTrading package hands you in the OA: you don't build the plumbing,
you consume typed objects and return typed objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Quote:
    """A two-sided price a market maker is willing to trade at."""

    bid: float
    ask: float

    def __post_init__(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"quote prices must be positive, got bid={self.bid}, ask={self.ask}")
        if self.bid >= self.ask:
            raise ValueError(f"crossed or locked quote: bid={self.bid} >= ask={self.ask}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        return 10_000 * self.spread / self.mid


@dataclass
class MarketState:
    """Everything a market maker's `quote()` is allowed to look at, at time t.

    time_remaining is in [0, 1]: 1.0 at the start of the trading horizon,
    0.0 at the close. It exists because inventory risk is not static -
    a market maker MUST be closer to flat near the end of its horizon than
    it needs to be at the start (see Day 2: Avellaneda-Stoikov).
    """

    timestamp: int
    ref_price: float
    realized_vol: float
    time_remaining: float
    inventory: float = 0.0


@dataclass(frozen=True)
class ExchangeTrade:
    """A trade instruction / confirmation moving through the mock Exchange."""

    ticker: str
    trade_volume: float
    ref_price: float
    action: Side
    timestamp: int

    def __post_init__(self) -> None:
        if self.trade_volume <= 0:
            raise ValueError("trade_volume must be positive; direction is carried by `action`")
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {self.action!r}")

    @property
    def notional(self) -> float:
        return self.trade_volume * self.ref_price

    @property
    def signed_volume(self) -> float:
        return self.trade_volume if self.action == "buy" else -self.trade_volume
