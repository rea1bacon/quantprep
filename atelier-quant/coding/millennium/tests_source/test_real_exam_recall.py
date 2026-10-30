"""Grading suite for the real-exam recall problems.

Run with:  pytest tests/test_real_exam_recall.py -v

Expected values were computed once by running
solutions/real_exam_recall_solution.py on the fixtures below and hardcoded,
the same way every other day's ground-truth checks work in this program.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import _mini_pytest as pytest


import _cible as mod
clean_prices = mod.clean_prices
liquidity_scores = mod.liquidity_scores
reference_price = mod.reference_price
quote_request = mod.quote_request
quote_impact_stream = mod.quote_impact_stream
track_book = mod.track_book


# ============================================================
# Problem 1 - Clean The Tape
# ============================================================
def test_clean_prices():
    raw = pd.DataFrame(
        [
            {"timestamp": "2026-01-01 09:40:00", "Asset_1": 103.0, "Asset_2": 52.0},
            {"timestamp": "2026-01-01 09:30:00", "Asset_1": 100.0, "Asset_2": -1.0},
            {"timestamp": "2026-01-01 09:30:00", "Asset_1": 100.0, "Asset_2": -1.0},  # exact duplicate
            {"timestamp": "2026-01-01 09:35:00", "Asset_1": 0.0, "Asset_2": 50.0},  # Asset_1 invalid
            {"timestamp": "2026-01-01 09:45:00", "Asset_1": 105.0, "Asset_2": -5.0},  # Asset_2 invalid
        ]
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    out = clean_prices(raw)

    assert len(out) == 4, "the exact duplicate row should have been dropped"
    assert list(out.index) == [0, 1, 2, 3], "index must be reset to 0..n-1"
    assert out["timestamp"].is_monotonic_increasing

    assert out["Asset_1"].tolist() == pytest.approx([100.0, 100.0, 103.0, 105.0])
    # Asset_2's first bar (09:30) was invalid with nothing earlier to ffill
    # from -> must stay NaN, not get silently zero-filled or dropped.
    assert pd.isna(out["Asset_2"].iloc[0]), "a leading gap must stay NaN, not be filled"
    assert out["Asset_2"].iloc[1:].tolist() == pytest.approx([50.0, 52.0, 52.0])


def test_clean_prices_does_not_mutate_the_caller():
    raw = pd.DataFrame(
        {"timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:35:00"], "A": [100.0, -1.0]}
    )
    before = raw.copy(deep=True)

    clean_prices(raw)

    # repair a COPY - the caller's tape must survive unchanged
    pd.testing.assert_frame_equal(raw, before)


def test_clean_prices_treats_existing_nans_like_bad_prints():
    # A already has NaNs (not just <= 0 values); B is invalid all the way down.
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 09:30:00", "2026-01-01 09:35:00", "2026-01-01 09:40:00"]
            ),
            "A": [np.nan, 101.0, np.nan],
            "B": [-1.0, 0.0, -2.0],
        }
    )

    out = clean_prices(raw)

    assert pd.isna(out["A"].iloc[0]), "a leading NaN has nothing to carry forward"
    assert out["A"].iloc[1:].tolist() == pytest.approx([101.0, 101.0])
    assert out["B"].isna().all(), "an asset that never printed a valid price stays all NaN"


def test_clean_prices_keeps_the_first_row_of_a_repeated_timestamp():
    # Two prints stamped the same instant: the tape keeps one bar per timestamp
    # (the first), rather than leaving an ambiguous double bar for the as-of
    # lookup in Problem 3 to trip over.
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:30:00", "2026-01-01 09:30:00"]),
            "A": [100.0, 101.0],
        }
    )

    out = clean_prices(raw)

    assert len(out) == 1
    assert out["A"].iloc[0] == pytest.approx(100.0)


def test_clean_prices_accepts_string_timestamps():
    # The spec allows a timestamp column pandas can parse, not only datetime64.
    raw = pd.DataFrame(
        {"timestamp": ["2026-01-01 09:40:00", "2026-01-01 09:30:00"], "A": [102.0, 100.0]}
    )

    out = clean_prices(raw)

    assert out["A"].tolist() == pytest.approx([100.0, 102.0]), "rows must end up in time order"



# ============================================================
# Problem 2 - Measure Liquidity
# ============================================================
def test_liquidity_scores_in_range_and_sane_ordering():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:30", periods=6, freq="5min"),
            "Asset_A": [100, 100.5, 100.2, 100.7, 100.4, 100.9],  # calm
            "Asset_B": [100, 102, 99, 103, 98, 104],  # medium vol
            "Asset_C": [100, 110, 90, 115, 85, 120],  # wild
        }
    )
    requests = pd.DataFrame({"asset": ["Asset_A"] * 2 + ["Asset_B"] * 5 + ["Asset_C"] * 10})

    scores = liquidity_scores(prices, requests)

    assert set(scores.keys()) == {"Asset_A", "Asset_B", "Asset_C"}
    for v in scores.values():
        assert 0.1 <= v <= 1.0, "scores must stay inside the [0.1, 1.0] floor/ceiling"

    expected = {"Asset_A": 0.55, "Asset_B": 0.652224, "Asset_C": 0.55}
    for asset, exp in expected.items():
        assert scores[asset] == pytest.approx(exp, abs=1e-6)


def test_liquidity_scores_zero_range_defaults_to_half():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"),
            "Asset_A": [100, 101, 100],
            "Asset_B": [50, 50, 50],
        }
    )
    empty_requests = pd.DataFrame({"asset": []})

    scores = liquidity_scores(prices, empty_requests)
    # With no requests at all, frequency is 0 for both assets -> zero range
    # -> normalized frequency must default to 0.5 for everyone, not crash
    # or divide by zero.
    assert scores == pytest.approx({"Asset_A": 0.325, "Asset_B": 0.775}, abs=1e-6)


def test_liquidity_scores_ignores_extra_request_columns():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "A": [100, 101, 100, 102.0],
            "B": [50, 55, 45, 60.0],
        }
    )
    bare = pd.DataFrame({"asset": ["A", "A", "B"]})
    padded = bare.assign(quantity=[5, 6, 7], side=["BUY", "SELL", "BUY"])

    assert liquidity_scores(prices, padded) == pytest.approx(liquidity_scores(prices, bare)), (
        "columns other than `asset` carry no information here and must be ignored"
    )


def test_liquidity_scores_are_rounded_to_six_decimals():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 09:30", periods=6, freq="5min"),
            "Asset_A": [100, 100.5, 100.2, 100.7, 100.4, 100.9],
            "Asset_B": [100, 102, 99, 103, 98, 104],
            "Asset_C": [100, 110, 90, 115, 85, 120],
        }
    )
    requests = pd.DataFrame({"asset": ["Asset_A"] * 2 + ["Asset_B"] * 5 + ["Asset_C"] * 10})

    for asset, score in liquidity_scores(prices, requests).items():
        assert score == round(score, 6), f"{asset} was not rounded to 6 decimals"



# ============================================================
# Problem 3 - Fair Value At A Time
# ============================================================
@pytest.fixture
def prices3():
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 09:40:00",
                "2026-01-01 09:30:00",
                "2026-01-01 09:35:00",
                "2026-01-01 09:50:00",
            ],
            "X": [102.0, 100.0, None, None],
        }
    )


def test_reference_price_exact_match(prices3):
    assert reference_price(prices3, "X", "2026-01-01 09:40:00", 5) == {"price": 102.0}


def test_reference_price_skips_nulls_and_within_staleness(prices3):
    # nearest bar at/before 09:37 is the null at 09:35 - must skip it and use 09:30 (100.0)
    assert reference_price(prices3, "X", "2026-01-01 09:37:00", 10) == {"price": 100.0}


def test_reference_price_stale_returns_none(prices3):
    # same lookup as above, but the 7-minute gap now exceeds the 5-minute budget
    assert reference_price(prices3, "X", "2026-01-01 09:37:00", 5) == {"price": None}


def test_reference_price_before_first_print_returns_none(prices3):
    assert reference_price(prices3, "X", "2026-01-01 09:25:00", 100) == {"price": None}


def test_reference_price_exactly_at_staleness_boundary_is_ok(prices3):
    # 09:44 is exactly 4 minutes after the 09:40 print; max_staleness=4 should
    # still be acceptable ("exactly max_staleness_minutes old is still ok").
    assert reference_price(prices3, "X", "2026-01-01 09:44:00", 4) == {"price": 102.0}


def test_reference_price_uses_most_recent_valid_not_highest_value():
    prices = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:40:00"],
            "X": [105.0, 100.0],
        }
    )
    # As-of logic must pick the last valid print at/before ts (100.0 at 09:40),
    # not the numerically largest historical value (105.0 at 09:30).
    assert reference_price(prices, "X", "2026-01-01 09:41:00", 10) == {"price": 100.0}


def test_reference_price_skips_non_positive_values():
    prices = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:35:00", "2026-01-01 09:40:00"],
            "X": [100.0, 0.0, -5.0],
        }
    )
    # 0 and negative values are invalid and must be skipped like NaN.
    assert reference_price(prices, "X", "2026-01-01 09:40:00", 20) == {"price": 100.0}


def test_reference_price_returns_none_if_no_valid_price_at_or_before_ts():
    prices = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30:00", "2026-01-01 09:35:00"],
            "X": [0.0, -1.0],
        }
    )
    assert reference_price(prices, "X", "2026-01-01 09:36:00", 20) == {"price": None}


def test_reference_price_staleness_counts_whole_days():
    # A print from yesterday is 25 hours old, not 1. Reading only the
    # seconds-within-the-day component of the gap (Timedelta.seconds instead of
    # .total_seconds()) makes an overnight or weekend gap look fresh.
    prices = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 09:30:00"]), "X": [100.0]}
    )

    assert reference_price(prices, "X", "2026-01-02 10:30:00", 60) == {"price": None}


def test_reference_price_ignores_other_assets_validity():
    # Asset_2 is broken on the 09:40 bar, but Asset_1 printed a perfectly good
    # price there. Discarding the whole row (a bare dropna) would fall back to
    # 09:30 and wrongly report the quote as stale.
    prices = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:30:00", "2026-01-01 09:40:00"]),
            "Asset_1": [100.0, 103.0],
            "Asset_2": [50.0, -1.0],
        }
    )

    assert reference_price(prices, "Asset_1", "2026-01-01 09:41:00", 5) == {"price": 103.0}


def test_reference_price_does_not_mutate_the_caller():
    prices = pd.DataFrame(
        {"timestamp": ["2026-01-01 09:30:00"], "X": [100.0]}
    )
    before = prices.copy(deep=True)

    reference_price(prices, "X", "2026-01-01 09:31:00", 5)

    # a lookup must not rewrite its input
    pd.testing.assert_frame_equal(prices, before)


def test_reference_price_accepts_a_timestamp_object(prices3):
    # `ts` may arrive already built, not only as a string.
    assert reference_price(prices3, "X", pd.Timestamp("2026-01-01 09:40:00"), 5) == {"price": 102.0}



# ============================================================
# Problem 4 - Quote One Request
# ============================================================
def test_quote_request_buy_and_sell():
    q_buy = quote_request(100.0, 0.5, "BUY", 13, {"k": 0.01}, 0.05)
    q_sell = quote_request(100.0, 0.5, "SELL", 13, {"k": 0.01}, 0.05)
    assert q_buy == pytest.approx(100.65)
    assert q_sell == pytest.approx(99.35)


def test_quote_request_lands_exactly_on_tick_grid():
    # reference + half_spread = 100.5, exactly on a 0.05 grid with qty=0 -> no
    # floating-point rounding should push this to 100.55 or 100.45.
    q = quote_request(100.0, 0.5, "BUY", 0, {"k": 0.01}, 0.05)
    assert q == pytest.approx(100.5)


def test_quote_request_bigger_size_worsens_price():
    small = quote_request(100.0, 0.5, "BUY", 5, {"k": 0.01}, 0.05)
    large = quote_request(100.0, 0.5, "BUY", 50, {"k": 0.01}, 0.05)
    assert large > small, "a bigger BUY quantity must push the quoted price up, not down"


def test_quote_request_sell_lands_exactly_on_tick_grid():
    # The BUY side rounds up and the SELL side rounds down, so each has its own
    # floating-point boundary to get wrong: 100.0 - 0.5 is exactly on the grid.
    q = quote_request(100.0, 0.5, "SELL", 0, {"k": 0.01}, 0.05)
    assert q == pytest.approx(99.5)



# ============================================================
# Problem 5 - Debug The Impact Model
# ============================================================
def test_quote_impact_stream_matches_expected_sequence():
    requests = [
        {"side": "BUY", "quantity": 10},
        {"side": "BUY", "quantity": 10},
        {"side": "SELL", "quantity": 5},
    ]
    quotes = quote_impact_stream(100.0, {"k": 0.02, "decay": 0.9, "half_spread": 0.1}, 0.5, requests)
    assert quotes == pytest.approx([100.1, 100.19, 100.071], abs=1e-6)


def test_quote_impact_stream_full_liquidity_means_no_impact():
    requests = [
        {"side": "BUY", "quantity": 10},
        {"side": "BUY", "quantity": 10},
        {"side": "SELL", "quantity": 5},
    ]
    quotes = quote_impact_stream(100.0, {"k": 0.02, "decay": 0.9, "half_spread": 0.1}, 1.0, requests)
    # liquidity_score=1.0 must zero out impact_term entirely - every quote is
    # just reference +/- half_spread, the state never moves.
    assert quotes == pytest.approx([100.1, 100.1, 99.9], abs=1e-6)


def test_quote_impact_stream_empty_requests():
    quotes = quote_impact_stream(100.0, {"k": 0.02, "decay": 0.9, "half_spread": 0.1}, 0.5, [])
    assert quotes == []


def test_quote_impact_stream_first_quote_excludes_its_own_impact():
    # liquidity_score=0 makes the impact as large as it gets, so adding this
    # order's impact before quoting it would be impossible to miss.
    quotes = quote_impact_stream(
        100.0, {"k": 0.02, "decay": 0.9, "half_spread": 0.1}, 0.0, [{"side": "BUY", "quantity": 10}]
    )
    assert quotes == pytest.approx([100.1]), "the first quote is just reference + half_spread"


def test_quote_impact_stream_decay_of_one_keeps_the_whole_state():
    requests = [
        {"side": "BUY", "quantity": 10},
        {"side": "BUY", "quantity": 10},
        {"side": "SELL", "quantity": 5},
    ]
    quotes = quote_impact_stream(100.0, {"k": 0.02, "decay": 1.0, "half_spread": 0.1}, 0.5, requests)
    # decay=1.0 means nothing reverts: the state accumulates 0.1 per buy and
    # the third quote still carries both of them.
    assert quotes == pytest.approx([100.1, 100.2, 100.1], abs=1e-6)



# ============================================================
# Problem 6 - Track The Book
# ============================================================
def test_track_book():
    prices = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"), "X": [100, 105], "Y": [50, 50]}
    )
    filled = pd.DataFrame(
        [
            {"asset": "X", "side": "BUY", "quantity": 10, "fill_price": 100},
            {"asset": "X", "side": "SELL", "quantity": 3, "fill_price": 102},
            {"asset": "Y", "side": "SELL", "quantity": 20, "fill_price": 48},
        ]
    )
    result = track_book(filled, prices)

    assert result["inventory"] == {"X": -7, "Y": 20}
    assert result["cash"] == pytest.approx(-266.0)
    assert result["pnl"] == pytest.approx(-1.0)


def test_track_book_empty_stays_flat():
    prices = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"), "X": [100, 105], "Y": [50, 50]}
    )
    empty = pd.DataFrame(columns=["asset", "side", "quantity", "fill_price"])

    result = track_book(empty, prices)
    assert result["inventory"] == {"X": 0, "Y": 0}
    assert result["cash"] == pytest.approx(0.0)
    assert result["pnl"] == pytest.approx(0.0)


def test_track_book_untraded_asset_stays_flat():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"),
            "X": [100.0, 105.0],
            "Z": [10.0, 11.0],
        }
    )
    filled = pd.DataFrame([{"asset": "X", "side": "BUY", "quantity": 10, "fill_price": 100.0}])

    result = track_book(filled, prices)

    assert result["inventory"] == {"X": -10, "Z": 0}, "every asset in the tape needs an entry"
    assert result["pnl"] == pytest.approx(-50.0), "Z contributes nothing while flat"
