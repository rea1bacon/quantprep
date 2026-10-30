"""Execution simulation: a slippage-aware Exchange, and a fill-probability
backtest engine for market makers.

The backtest engine simulates counterparties hitting your quotes according to
how close they are to fair value, and marks your resulting inventory to market
every tick. Fills do NOT depend on the next price move: this engine contains no
informed flow, hence no adverse selection (corrected from the original wording).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .base import MarketMakerBase
from .types import ExchangeTrade, MarketState


class Exchange:
    """A minimal execution venue: fills at ref_price plus/minus slippage."""

    def __init__(self, slippage_bps: float = 0.0):
        self.slippage_bps = slippage_bps
        self.trade_log: list[ExchangeTrade] = []

    def execute(self, trade: ExchangeTrade) -> ExchangeTrade:
        slip = trade.ref_price * self.slippage_bps / 10_000
        filled_price = trade.ref_price + slip if trade.action == "buy" else trade.ref_price - slip
        filled = replace(trade, ref_price=round(filled_price, 4))
        self.trade_log.append(filled)
        return filled

    def execute_with_latency(
        self,
        trade: ExchangeTrade,
        latency_ticks: int,
        price_path: np.ndarray,
        current_tick: int,
    ) -> ExchangeTrade:
        """Fill a trade `latency_ticks` after it was decided, at whatever the
        price has drifted to by then. This is the 'legging risk' mechanic
        used in Day 4: the first leg of an arbitrage fills instantly, later
        legs fill after a delay during which the market can move against you.
        """
        fill_tick = min(current_tick + latency_ticks, len(price_path) - 1)
        drifted_price = float(price_path[fill_tick])
        delayed = replace(trade, ref_price=round(drifted_price, 4), timestamp=fill_tick)
        return self.execute(delayed)


def simulate_market_making(
    mm: MarketMakerBase,
    fair_value: np.ndarray,
    dt: float = 1.0,
    base_intensity: float = 0.35,
    fill_decay: float = 140.0,
    max_inventory: float | None = None,
    vol_window: int = 30,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run `mm` tick-by-tick against a fair-value path and return the P&L trace.

    Fill mechanics: at each tick, a counterparty may lift the ask or hit the
    bid. The probability of each decays exponentially with how far that quote
    sits from fair value (a quote right on top of fair value is very likely to
    trade; a quote 3% away almost never is) - this is what makes wide,
    volatility-scaled spreads a real trade-off, not a free lunch: wider
    spreads earn more per fill but get fewer fills. base_intensity is a
    per-tick fill probability scale and is intentionally NOT rescaled by dt -
    it calibrates "how liquid is one tick of this simulated market", which is
    independent of how many calendar-years one tick happens to represent.

    max_inventory, if set, hard-stops further same-direction fills once the
    position limit is breached - your MM must actively skew back inside the
    limit, it cannot just keep quoting symmetrically and rely on luck.

    dt is used ONLY to annualize realized_vol and must be in the SAME units
    used to generate `fair_value` (e.g. pass day.attrs["dt"] from
    generate_trading_day) - otherwise the realized_vol handed to your
    MarketMaker will be on the wrong scale and every vol-sensitive quoting
    decision downstream will be silently wrong. Do not conflate this with
    fill-probability calibration; the two are deliberately independent.
    """
    if len(fair_value) < 2:
        raise ValueError("fair_value path must have at least 2 points")
    rng = np.random.default_rng(seed)
    n = len(fair_value)
    log_returns = np.diff(np.log(fair_value), prepend=np.log(fair_value[0]))

    inventory = 0.0
    cash = 0.0
    rows = []

    for t in range(n):
        window = log_returns[max(0, t - vol_window) : t + 1]
        realized_vol = float(window.std() * np.sqrt(1.0 / dt)) if len(window) > 1 else 0.0

        state = MarketState(
            timestamp=t,
            ref_price=float(fair_value[t]),
            realized_vol=realized_vol,
            time_remaining=1.0 - t / (n - 1),
            inventory=inventory,
        )
        quote = mm.quote(state)

        p_ask_hit = base_intensity * np.exp(-fill_decay * (quote.ask - fair_value[t]) / fair_value[t])
        p_bid_hit = base_intensity * np.exp(-fill_decay * (fair_value[t] - quote.bid) / fair_value[t])
        p_ask_hit = float(np.clip(p_ask_hit, 0.0, 1.0))
        p_bid_hit = float(np.clip(p_bid_hit, 0.0, 1.0))

        if (max_inventory is None or inventory > -max_inventory) and rng.random() < p_ask_hit:
            inventory -= 1
            cash += quote.ask
            mm.on_fill("sell", 1.0, quote.ask)

        if (max_inventory is None or inventory < max_inventory) and rng.random() < p_bid_hit:
            inventory += 1
            cash -= quote.bid
            mm.on_fill("buy", 1.0, quote.bid)

        mtm_pnl = cash + inventory * fair_value[t]
        rows.append(
            dict(
                t=t,
                fair=fair_value[t],
                bid=quote.bid,
                ask=quote.ask,
                spread_bps=quote.spread_bps,
                inventory=inventory,
                cash=cash,
                mtm_pnl=mtm_pnl,
            )
        )

    return pd.DataFrame(rows)
