"""Performance evaluation primitives.

These are the kind of ready-made scoring utilities a real framework gives
you - estimation techniques (beta, cointegration, half-life) are deliberately
NOT here, because building those is the point of the exercises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of a returns series. NaN if the series has no variance."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    excess = r - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / r.std(ddof=1))


def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough decline, expressed as a negative fraction (e.g. -0.23)."""
    e = np.asarray(equity_curve, dtype=float)
    if len(e) < 2:
        return 0.0
    running_max = np.maximum.accumulate(e)
    drawdown = (e - running_max) / np.where(running_max == 0, np.nan, running_max)
    return float(np.nanmin(drawdown))


def hedge_effectiveness(hedged_returns: pd.Series | np.ndarray, unhedged_returns: pd.Series | np.ndarray) -> float:
    """Fraction of variance removed by hedging: 1 - var(hedged)/var(unhedged).

    1.0 = perfect hedge, 0.0 = hedge did nothing, negative = hedge made it worse
    (e.g. wrong sign, or rebalanced so infrequently that basis risk dominates).
    """
    h = np.asarray(hedged_returns, dtype=float)
    u = np.asarray(unhedged_returns, dtype=float)
    var_u = np.nanvar(u)
    if var_u == 0:
        return float("nan")
    return float(1 - np.nanvar(h) / var_u)


def inventory_stats(inventory: pd.Series | np.ndarray) -> dict[str, float]:
    """Summary stats a risk desk would actually look at, not just mean P&L."""
    inv = np.asarray(inventory, dtype=float)
    return {
        "mean_abs_inventory": float(np.mean(np.abs(inv))),
        "max_abs_inventory": float(np.max(np.abs(inv))),
        "inventory_std": float(np.std(inv)),
        "time_flat_pct": float(np.mean(inv == 0) * 100),
    }
