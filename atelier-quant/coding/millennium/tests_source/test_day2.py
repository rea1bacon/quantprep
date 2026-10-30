"""Hidden-test-style grading suite for Day 2.

Run with:  pytest tests/test_day2.py -v
"""
from __future__ import annotations

import numpy as np
import _mini_pytest as pytest

from amplify_quant_trading import (
    MarketMakerBase,
    MarketState,
    Quote,
    generate_trading_day,
    simulate_market_making,
)

import _cible as mod
AvellanedaStoikovMarketMaker = mod.AvellanedaStoikovMarketMaker


class NaiveFixedSpreadMM(MarketMakerBase):
    """Same 1.5%-each-side rule as Millennium's real Challenge 1 - the baseline
    every inventory-aware model in this program is measured against."""

    def quote(self, state: MarketState) -> Quote:
        return Quote(bid=round(state.ref_price * 0.985, 2), ask=round(state.ref_price * 1.015, 2))


def make_state(ref_price=100.0, realized_vol=0.3, time_remaining=1.0, inventory=0.0):
    return MarketState(
        timestamp=0,
        ref_price=ref_price,
        realized_vol=realized_vol,
        time_remaining=time_remaining,
        inventory=inventory,
    )


def test_quote_is_valid():
    mm = AvellanedaStoikovMarketMaker()
    q = mm.quote(make_state(inventory=5))
    assert q.bid > 0
    assert q.ask > q.bid


def test_zero_inventory_has_no_skew():
    mm = AvellanedaStoikovMarketMaker()
    q = mm.quote(make_state(ref_price=100.0, inventory=0.0))
    assert q.mid == pytest.approx(100.0, abs=0.05)


def test_skews_down_when_long():
    mm = AvellanedaStoikovMarketMaker()
    q = mm.quote(make_state(ref_price=100.0, inventory=5.0))
    assert q.mid < 100.0


def test_skews_up_when_short():
    mm = AvellanedaStoikovMarketMaker()
    q = mm.quote(make_state(ref_price=100.0, inventory=-5.0))
    assert q.mid > 100.0


def test_skew_magnitude_grows_with_inventory():
    mm = AvellanedaStoikovMarketMaker()
    small = mm.quote(make_state(ref_price=100.0, inventory=2.0))
    large = mm.quote(make_state(ref_price=100.0, inventory=8.0))
    skew_small = 100.0 - small.mid
    skew_large = 100.0 - large.mid
    assert skew_large > skew_small > 0


def test_extreme_inventory_never_produces_an_invalid_quote():
    mm = AvellanedaStoikovMarketMaker()
    for inv in (1_000.0, 100_000.0, -100_000.0):
        q = mm.quote(make_state(ref_price=100.0, inventory=inv))
        assert q.bid > 0
        assert q.ask > q.bid


def test_spread_still_widens_with_volatility():
    mm = AvellanedaStoikovMarketMaker()
    calm = mm.quote(make_state(realized_vol=0.05, inventory=0.0))
    stressed = mm.quote(make_state(realized_vol=0.60, inventory=0.0))
    assert stressed.spread_bps > calm.spread_bps


def test_backtest_reduces_inventory_variance_vs_naive_fixed_spread():
    day = generate_trading_day(n_ticks=5000, seed=23)
    kwargs = dict(dt=day.attrs["dt"], seed=31, max_inventory=60)

    as_result = simulate_market_making(AvellanedaStoikovMarketMaker(), day["price"].to_numpy(), **kwargs)
    naive_result = simulate_market_making(NaiveFixedSpreadMM(), day["price"].to_numpy(), **kwargs)

    as_inv_std = as_result["inventory"].std()
    naive_inv_std = naive_result["inventory"].std()

    assert as_inv_std < naive_inv_std * 0.85, (
        "an inventory-aware market maker should hold a measurably less volatile position than "
        f"a fixed-spread one on the same price path - got AS std={as_inv_std:.2f}, "
        f"naive std={naive_inv_std:.2f}"
    )


def test_pnl_stays_finite():
    day = generate_trading_day(n_ticks=3000, seed=4)
    mm = AvellanedaStoikovMarketMaker()
    result = simulate_market_making(mm, day["price"].to_numpy(), dt=day.attrs["dt"], seed=6, max_inventory=40)
    assert np.isfinite(result["mtm_pnl"]).all()
