"""Hidden-test-style grading suite for Day 4.

Run with:  pytest tests/test_day4.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import _mini_pytest as pytest

from amplify_quant_trading import generate_etf_basket

import _cible as mod
CointegrationArbBot = mod.CointegrationArbBot
estimate_half_life = mod.estimate_half_life


def test_synthetic_price_is_weighted_sum():
    bot = CointegrationArbBot()
    weights = {"A": 0.4, "B": 0.6}
    prices = {"A": 10.0, "B": 20.0}
    assert bot.synthetic_price(weights, prices) == pytest.approx(16.0)


def test_zscore_matches_hand_computation():
    bot = CointegrationArbBot()
    history = pd.Series([0.0, 0.0, 0.0, 0.0, 0.02])
    # mean=0.004, population std=0.008 -> z = (0.02-0.004)/0.008 = 2.0
    assert bot.zscore(history) == pytest.approx(2.0, rel=1e-3)


def test_signal_no_trade_when_zscore_small():
    bot = CointegrationArbBot(entry_z=2.0, commission_pct=0.001, min_edge_multiple=1.2)
    assert bot.signal(mispricing_pct=0.05, z_score=0.5) == "no_trade"


def test_signal_no_trade_when_edge_below_commission_floor():
    bot = CointegrationArbBot(entry_z=2.0, commission_pct=0.001, min_edge_multiple=1.2)
    # floor = 1.2 * (2*0.001) = 0.0024; 0.001 < floor even though z is large
    assert bot.signal(mispricing_pct=0.001, z_score=3.0) == "no_trade"


def test_signal_sell_etf_when_rich():
    bot = CointegrationArbBot(entry_z=2.0, commission_pct=0.001, min_edge_multiple=1.2)
    assert bot.signal(mispricing_pct=0.01, z_score=3.0) == "sell_etf_buy_basket"


def test_signal_buy_etf_when_cheap():
    bot = CointegrationArbBot(entry_z=2.0, commission_pct=0.001, min_edge_multiple=1.2)
    assert bot.signal(mispricing_pct=-0.01, z_score=-3.0) == "buy_etf_sell_basket"


def test_half_life_recovers_planted_value():
    basket, weights = generate_etf_basket(n_days=3000, n_components=4, seed=11)
    mispricing = basket["ETF"] / basket["synthetic_true"] - 1

    half_life_hat = estimate_half_life(mispricing)
    true_half_life = basket.attrs["ou_half_life_days"]

    assert half_life_hat == pytest.approx(true_half_life, rel=0.35), (
        f"estimated half-life {half_life_hat:.1f} days vs true {true_half_life:.1f} days - "
        "check your AR(1) fit (are x_t and x_{t-1} aligned correctly?)"
    )


def test_strategy_is_net_profitable_after_realistic_costs():
    basket, weights = generate_etf_basket(n_days=3000, n_components=4, seed=11)
    etf = basket["ETF"].to_numpy()
    synth = basket["synthetic_true"].to_numpy()
    mispricing_pct = etf / synth - 1

    # 2bps/leg is aggressive but realistic for a top-tier desk's execution
    # cost on a liquid, large-cap ETF/basket pair. min_edge_multiple=4.0
    # keeps the entry bar comfortably above the FULL round-trip cost (entry
    # + exit, 4 legs) - not just the entry-leg cost - which is the whole
    # "commission drag" lesson from the theory doc: a mispricing that only
    # clears the cost of getting IN is not automatically a profitable trade.
    bot = CointegrationArbBot(entry_z=2.0, commission_pct=0.0002, min_edge_multiple=4.0)
    window = 60
    max_hold_days = 40  # a mean-reverting position that never reverts still needs a stop
    exit_fraction = 0.3  # exit once 70% of the entry edge has closed
    round_trip_cost = 2 * (2 * bot.commission_pct)  # 2 legs to enter, 2 legs to exit

    in_position = False
    entry_edge = 0.0
    entry_t = 0
    total_pnl_pct = 0.0
    n_trades = 0

    for t in range(window, len(mispricing_pct)):
        hist = pd.Series(mispricing_pct[t - window : t + 1])
        z = bot.zscore(hist)
        m_pct = mispricing_pct[t]

        if not in_position:
            sig = bot.signal(m_pct, z)
            if sig != "no_trade":
                in_position = True
                entry_edge = abs(m_pct)
                entry_t = t
                n_trades += 1
        else:
            held = t - entry_t
            if abs(m_pct) <= entry_edge * exit_fraction or held >= max_hold_days:
                exit_edge = abs(m_pct)
                total_pnl_pct += (entry_edge - exit_edge) - round_trip_cost
                in_position = False

    assert n_trades >= 3, (
        f"strategy only found {n_trades} trades over {len(mispricing_pct)} days - "
        "thresholds are probably too strict to ever fire"
    )
    assert total_pnl_pct > 0, f"strategy lost money net of round-trip commissions: {total_pnl_pct:.5f}"
