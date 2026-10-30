"""Abstract interfaces you implement.

These are deliberately shaped like "given stub" interfaces from a hiring
assessment: a class with abstract methods and a docstring contract. Nothing
here does the job for you - the exercises/ files subclass these.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .types import ExchangeTrade, MarketState, Quote, Side


class MarketMakerBase(ABC):
    """Contract for anything that quotes a two-sided market.

    quote() is called once per tick with the current MarketState (which
    already carries this MM's own inventory, injected by the backtest
    engine). It must be a pure function of `state` and `self` - no hidden
    randomness, no I/O.
    """

    @abstractmethod
    def quote(self, state: MarketState) -> Quote:
        raise NotImplementedError

    def on_fill(self, side: Side, volume: float, price: float) -> None:
        """Optional hook: called by the backtest engine when a quote gets hit/lifted.

        Override this if your quoting logic needs to react to fills beyond
        what the engine already tracks in MarketState.inventory (e.g. to
        maintain your own risk-limit state machine). Default is a no-op.
        """
        return None


class HedgerBase(ABC):
    """Contract for something that estimates and executes a hedge ratio."""

    @abstractmethod
    def hedge_ratio(self, asset_returns: pd.Series, hedge_returns: pd.Series) -> float:
        """Return the beta of `asset_returns` on `hedge_returns`."""
        raise NotImplementedError

    @abstractmethod
    def rebalance(
        self,
        equity_value: float,
        current_hedge_notional: float,
        target_ratio: float,
        hedge_price: float,
        timestamp: int = 0,
    ) -> Optional[ExchangeTrade]:
        """Return the trade needed to move the hedge to target, or None if inside the no-trade band."""
        raise NotImplementedError


class ArbitrageBotBase(ABC):
    """Contract for an ETF-vs-basket relative value (statistical arbitrage) bot."""

    @abstractmethod
    def synthetic_price(self, weights: dict[str, float], prices: dict[str, float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def zscore(self, mispricing_history: pd.Series) -> float:
        """Return the most recent mispricing's z-score against its own rolling history."""
        raise NotImplementedError

    @abstractmethod
    def signal(self, mispricing_pct: float, z_score: float) -> str:
        """Return one of 'sell_etf_buy_basket', 'buy_etf_sell_basket', 'no_trade'.

        Must consider BOTH: is the mispricing statistically unusual (z_score),
        and is it big enough in absolute terms to survive round-trip trading
        costs (mispricing_pct) - a statistically significant but tiny edge is
        still not a trade.
        """
        raise NotImplementedError
