"""Hidden-test-style grading suite for Day 1.

Run with:  pytest tests/test_day1.py -v
"""
from __future__ import annotations

import numpy as np
import _mini_pytest as pytest

from amplify_quant_trading import MarketState, generate_trading_day, simulate_market_making

import _cible as mod
VolAwareMarketMaker = mod.VolAwareMarketMaker


def make_state(ref_price=100.0, realized_vol=0.0, time_remaining=1.0, inventory=0.0):
    return MarketState(
        timestamp=0,
        ref_price=ref_price,
        realized_vol=realized_vol,
        time_remaining=time_remaining,
        inventory=inventory,
    )


def test_quote_is_valid():
    mm = VolAwareMarketMaker()
    q = mm.quote(make_state(realized_vol=0.2))
    assert q.bid > 0
    assert q.ask > q.bid


def test_zero_vol_still_has_positive_spread():
    mm = VolAwareMarketMaker()
    q = mm.quote(make_state(realized_vol=0.0))
    assert q.spread > 0


def test_spread_widens_with_volatility():
    mm = VolAwareMarketMaker()
    calm = mm.quote(make_state(realized_vol=0.05))
    stressed = mm.quote(make_state(realized_vol=0.60))
    assert stressed.spread_bps > calm.spread_bps


def test_deterministic_pure_function():
    mm = VolAwareMarketMaker()
    s = make_state(realized_vol=0.3)
    assert mm.quote(s) == mm.quote(s)


def test_spread_is_capped_under_extreme_vol():
    mm = VolAwareMarketMaker(max_spread_bps=600.0)
    q = mm.quote(make_state(realized_vol=5.0))  # an absurd volatility input
    assert q.spread_bps <= 600.0 + 1e-6


@pytest.mark.parametrize("ref_price", [10.0, 55.5, 1000.0])
def test_quote_brackets_ref_price(ref_price):
    mm = VolAwareMarketMaker()
    q = mm.quote(make_state(ref_price=ref_price, realized_vol=0.2))
    assert q.bid < ref_price < q.ask


def test_backtest_widens_spread_during_stress_regime():
    day = generate_trading_day(n_ticks=4000, seed=11)
    mm = VolAwareMarketMaker()
    result = simulate_market_making(
        mm, day["price"].to_numpy(), dt=day.attrs["dt"], seed=99, max_inventory=40
    )

    start, end = day.attrs["stress_window"]
    calm_mask = np.ones(len(result), dtype=bool)
    calm_mask[start:end] = False

    calm_spread = result.loc[calm_mask, "spread_bps"].mean()
    stress_spread = result.loc[~calm_mask, "spread_bps"].mean()
    assert stress_spread > calm_spread * 1.15, (
        "your realized-vol estimate should push spreads meaningfully wider during "
        "the stress window than during the calm window - got calm="
        f"{calm_spread:.1f}bps, stress={stress_spread:.1f}bps"
    )


def test_backtest_produces_realistic_fill_activity():
    """A market maker that never gets filled isn't a market maker - if your
    spread is enormous by default, or your Quote object is somehow rejecting
    trades, this will catch it even though `mtm_pnl` would technically still
    be 'finite' at exactly zero.
    """
    day = generate_trading_day(n_ticks=3000, seed=17)
    mm = VolAwareMarketMaker()
    result = simulate_market_making(
        mm, day["price"].to_numpy(), dt=day.attrs["dt"], seed=21, max_inventory=60
    )
    n_fills = (result["inventory"].diff().fillna(0) != 0).sum()
    assert n_fills > len(result) * 0.01, (
        f"only {n_fills} fills out of {len(result)} ticks - your spread is probably far "
        "too wide by default for any counterparty to ever trade against it"
    )


def test_pnl_and_inventory_stay_finite():
    day = generate_trading_day(n_ticks=3000, seed=3)
    mm = VolAwareMarketMaker()
    result = simulate_market_making(mm, day["price"].to_numpy(), seed=5, max_inventory=40)
    assert np.isfinite(result["mtm_pnl"]).all()
    assert np.isfinite(result["inventory"]).all()
