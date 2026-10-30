"""Hidden-test-style grading suite for Day 3.

Run with:  pytest tests/test_day3.py -v
"""
from __future__ import annotations

import numpy as np
import _mini_pytest as pytest

from amplify_quant_trading import generate_stock_etf_pair, hedge_effectiveness

import _cible as mod
RollingHedger = mod.RollingHedger


def test_hedge_ratio_recovers_true_beta():
    pair = generate_stock_etf_pair(n_days=1500, beta=0.72, seed=2)
    returns = pair[["etf", "stock"]].pct_change().dropna()

    hedger = RollingHedger()
    beta_hat = hedger.hedge_ratio(returns["stock"], returns["etf"])

    assert beta_hat == pytest.approx(pair.attrs["true_beta"], abs=0.12)


def test_hedge_ratio_handles_a_different_planted_beta():
    pair = generate_stock_etf_pair(n_days=1500, beta=1.35, seed=9)
    returns = pair[["etf", "stock"]].pct_change().dropna()

    hedger = RollingHedger()
    beta_hat = hedger.hedge_ratio(returns["stock"], returns["etf"])

    assert beta_hat == pytest.approx(pair.attrs["true_beta"], abs=0.15)


def test_rebalance_returns_none_inside_no_trade_band():
    hedger = RollingHedger(no_trade_band_pct=0.03)
    # target = -100_000 * 0.70 = -70_000; current = -69_000; drift = 1_000 = 1% of equity < 3% band
    trade = hedger.rebalance(
        equity_value=100_000, current_hedge_notional=-69_000, target_ratio=0.70, hedge_price=50.0
    )
    assert trade is None


def test_rebalance_returns_trade_outside_no_trade_band():
    hedger = RollingHedger(no_trade_band_pct=0.03)
    # target = -100_000 * 0.70 = -70_000; current = 0; drift = -70_000, way outside band
    trade = hedger.rebalance(
        equity_value=100_000, current_hedge_notional=0.0, target_ratio=0.70, hedge_price=50.0
    )
    assert trade is not None
    assert trade.action == "sell"
    assert trade.trade_volume == pytest.approx(70_000 / 50.0, rel=1e-6)


def test_rebalance_direction_is_buy_when_reducing_short():
    hedger = RollingHedger(no_trade_band_pct=0.03)
    # target = -100_000 * 0.70 = -70_000; current = -95_000 (over-hedged); need to BUY back 25_000
    trade = hedger.rebalance(
        equity_value=100_000, current_hedge_notional=-95_000, target_ratio=0.70, hedge_price=25.0
    )
    assert trade is not None
    assert trade.action == "buy"
    assert trade.trade_volume == pytest.approx(25_000 / 25.0, rel=1e-6)


def test_rebalance_rejects_non_positive_price():
    hedger = RollingHedger()
    with pytest.raises(ValueError):
        hedger.rebalance(equity_value=100_000, current_hedge_notional=0.0, target_ratio=0.7, hedge_price=0.0)


def test_rolling_hedge_reduces_return_variance_substantially():
    # Deliberately low idiosyncratic vol relative to etf_vol: this stock's
    # returns are mostly explained by the market factor, so a correctly
    # estimated hedge should remove most of its variance. (A stock that's
    # 90% idiosyncratic noise can never be meaningfully hedged against a
    # broad ETF - that's a property of the asset, not a bug in your hedger.)
    pair = generate_stock_etf_pair(n_days=1500, beta=0.9, idio_vol=0.12, seed=5)
    returns = pair[["etf", "stock"]].pct_change().dropna()

    hedger = RollingHedger(window=60)
    hedged = []
    unhedged = []
    for t in range(hedger.window, len(returns)):
        past_stock = returns["stock"].iloc[t - hedger.window : t]
        past_etf = returns["etf"].iloc[t - hedger.window : t]
        beta_hat = hedger.hedge_ratio(past_stock, past_etf)

        stock_ret_t = returns["stock"].iloc[t]
        etf_ret_t = returns["etf"].iloc[t]
        unhedged.append(stock_ret_t)
        hedged.append(stock_ret_t - beta_hat * etf_ret_t)

    effectiveness = hedge_effectiveness(np.array(hedged), np.array(unhedged))
    assert effectiveness > 0.5, f"expected the rolling hedge to remove >50% of variance, got {effectiveness:.2f}"
