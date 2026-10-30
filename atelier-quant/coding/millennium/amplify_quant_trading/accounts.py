"""Account bookkeeping shared by the hedging and arbitrage exercises."""
from __future__ import annotations

from .types import Side


class HedgeFund:
    """Tracks cash balance and per-ticker positions across multi-leg trades.

    This is intentionally dumb: it does no risk logic, no signal generation.
    It exists so your Day 3/4 strategy code has somewhere real to book trades,
    the same separation of concerns AmplifyQuantTrading's HedgeFund object
    gives you in the real assessment.
    """

    def __init__(self, balance: float, commission_pct: float = 0.001):
        if commission_pct < 0:
            raise ValueError("commission_pct cannot be negative")
        self.balance = balance
        self.commission_pct = commission_pct
        self.positions: dict[str, float] = {}

    def execute_order(self, ticker: str, volume: float, price: float, action: Side) -> None:
        if volume <= 0:
            raise ValueError("volume must be positive")
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")

        notional = volume * price
        commission = notional * self.commission_pct
        sign = 1 if action == "buy" else -1

        self.balance -= sign * notional + commission
        self.positions[ticker] = self.positions.get(ticker, 0.0) + sign * volume

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Total equity: cash balance plus the market value of every open position."""
        position_value = sum(qty * prices[ticker] for ticker, qty in self.positions.items() if qty != 0)
        return self.balance + position_value
