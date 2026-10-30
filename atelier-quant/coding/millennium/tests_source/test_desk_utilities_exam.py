"""Grading suite for the second desk-utilities practice set.

Run with:  pytest tests/test_desk_utilities_exam.py -v

Expected values were computed once by running
solutions/desk_utilities_exam_solution.py on the fixtures below and
hardcoded, the same way every other ground-truth check works in this program.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import _mini_pytest as pytest


import _cible as mod
reconcile_fills = mod.reconcile_fills
execution_quality_scores = mod.execution_quality_scores
batch_reference_prices = mod.batch_reference_prices
enrich_fills = mod.enrich_fills
bar_summary = mod.bar_summary
track_book_with_realized = mod.track_book_with_realized
daily_scorecard = mod.daily_scorecard


# ============================================================
# Problem 1 - Reconcile Trade Confirmations
# ============================================================
def test_reconcile_fills():
    conf = pd.DataFrame(
        [
            {"trade_id": "T1", "timestamp": "2026-01-01 09:35", "asset": "X", "price": 100.0, "quantity": 10, "status": "FILLED"},
            {"trade_id": "T1", "timestamp": "2026-01-01 09:35", "asset": "X", "price": 100.0, "quantity": 10, "status": "FILLED"},  # dup trade_id
            {"trade_id": "T2", "timestamp": "2026-01-01 09:30", "asset": "X", "price": -5.0, "quantity": 5, "status": "FILLED"},  # bad price
            {"trade_id": "T3", "timestamp": "2026-01-01 09:40", "asset": "Y", "price": 50.0, "quantity": 0, "status": "FILLED"},  # bad qty
            {"trade_id": "T4", "timestamp": "2026-01-01 09:32", "asset": "Y", "price": 51.0, "quantity": 8, "status": "CANCELLED"},
            {"trade_id": "T5", "timestamp": "2026-01-01 09:45", "asset": "Y", "price": 52.0, "quantity": 12, "status": "FILLED"},
        ]
    )
    out = reconcile_fills(conf)

    assert list(out["trade_id"]) == ["T1", "T5"]
    assert list(out.index) == [0, 1]
    assert (out["price"] > 0).all() and (out["quantity"] > 0).all()


def test_reconcile_fills_keeps_the_earliest_of_a_repeated_trade_id():
    # A resent confirmation carries the same trade_id at a later time. Sorting
    # by timestamp BEFORE deduplicating is what makes "keep the first" mean
    # "keep the original", not "keep whichever row happened to come first".
    conf = pd.DataFrame(
        [
            {"trade_id": "T1", "timestamp": "2026-01-01 09:45", "asset": "X", "price": 101.0, "quantity": 10, "status": "FILLED"},
            {"trade_id": "T1", "timestamp": "2026-01-01 09:30", "asset": "X", "price": 100.0, "quantity": 10, "status": "FILLED"},
        ]
    )

    out = reconcile_fills(conf)

    assert len(out) == 1
    assert out["price"].iloc[0] == pytest.approx(100.0), "the 09:30 original, not the 09:45 resend"


def test_reconcile_fills_all_rows_invalid_returns_an_empty_frame():
    conf = pd.DataFrame(
        [
            {"trade_id": "T9", "timestamp": "2026-01-01 09:30", "asset": "X", "price": -1.0, "quantity": 1, "status": "FILLED"},
            {"trade_id": "T8", "timestamp": "2026-01-01 09:31", "asset": "X", "price": 10.0, "quantity": 1, "status": "CANCELLED"},
        ]
    )

    out = reconcile_fills(conf)

    assert len(out) == 0
    assert list(out.columns) == list(conf.columns), "an empty result still keeps the schema"



# ============================================================
# Problem 2 - Rank Execution Quality
# ============================================================
def test_execution_quality_scores():
    fills = pd.DataFrame(
        {
            "asset": ["A"] * 3 + ["B"] * 3,
            "slippage_bps": [1, 2, 0, 10, 12, 8],
            "fill_time_ms": [50, 60, 40, 200, 220, 180],
        }
    )
    scores = execution_quality_scores(fills)
    # A has both lower slippage AND lower latency than B -> A should score
    # near the top of the range, B near the bottom.
    assert scores == pytest.approx({"A": 1.0, "B": 0.1}, abs=1e-6)


def test_execution_quality_scores_zero_range_defaults_to_half():
    # One asset means both signals have zero spread across assets; min-max
    # normalization is undefined, so each normalized signal is 0.5 ->
    # 0.1 + 0.9 * (0.5*0.5 + 0.5*0.5) = 0.55.
    fills = pd.DataFrame(
        {"asset": ["A", "A"], "slippage_bps": [5, 5], "fill_time_ms": [100, 100]}
    )
    assert execution_quality_scores(fills) == pytest.approx({"A": 0.55}, abs=1e-6)


def test_execution_quality_scores_uses_the_mean_not_the_total():
    # A traded six times, all of them good; B traded once, badly. Summing
    # instead of averaging would make the busy, well-executed asset look like
    # the worst one on the desk.
    fills = pd.DataFrame(
        {
            "asset": ["A"] * 6 + ["B"],
            "slippage_bps": [1] * 6 + [9],
            "fill_time_ms": [10] * 6 + [90],
        }
    )
    scores = execution_quality_scores(fills)
    assert scores["A"] > scores["B"], "six good fills must not be penalised for being six"
    assert scores == pytest.approx({"A": 1.0, "B": 0.1}, abs=1e-6)



# ============================================================
# Problem 3 - Batch Fair Value Lookup
# ============================================================
def test_batch_reference_prices():
    prices_long = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 09:30", "2026-01-01 09:35", "2026-01-01 09:40",
                "2026-01-01 09:30", "2026-01-01 09:40",
            ],
            "asset": ["X", "X", "X", "Y", "Y"],
            "price": [100.0, None, 102.0, 50.0, 53.0],
        }
    )
    queries = pd.DataFrame(
        {
            "asset": ["X", "X", "Y", "X"],
            "ts": ["2026-01-01 09:40:00", "2026-01-01 09:37:00", "2026-01-01 09:25:00", "2026-01-01 09:37:00"],
        }
    )

    result_generous = batch_reference_prices(prices_long, queries, max_staleness_minutes=10)
    assert result_generous == [102.0, 100.0, None, 100.0]

    result_strict = batch_reference_prices(prices_long, queries, max_staleness_minutes=6)
    assert result_strict == [102.0, None, None, None]


def test_batch_reference_prices_empty_queries():
    prices_long = pd.DataFrame(
        {"timestamp": ["2026-01-01 09:30"], "asset": ["X"], "price": [100.0]}
    )
    assert batch_reference_prices(prices_long, pd.DataFrame({"asset": [], "ts": []}), 10) == []


def test_batch_reference_prices_staleness_boundary_is_inclusive():
    prices_long = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 09:30", "2026-01-01 09:40"],
            "asset": ["X", "X"],
            "price": [100.0, None],
        }
    )
    queries = pd.DataFrame({"asset": ["X"], "ts": ["2026-01-01 09:37:00"]})

    # The freshest valid print is 7 minutes old: 7 is acceptable, 6.9 is not.
    assert batch_reference_prices(prices_long, queries, 7) == [100.0]
    assert batch_reference_prices(prices_long, queries, 6.9) == [None]



# ============================================================
# Problem 4 - Join The Static Data
# ============================================================
@pytest.fixture
def instruments():
    inst = pd.DataFrame(
        [
            # X was republished: the 01-02 row is the one that counts
            {"asset": "X", "multiplier": 2.0, "currency": "USD", "as_of": "2026-01-02 08:00"},
            {"asset": "X", "multiplier": 1.0, "currency": "USD", "as_of": "2026-01-01 08:00"},
            {"asset": "Y", "multiplier": 5.0, "currency": "EUR", "as_of": "2026-01-01 08:00"},
        ]
    )
    inst["as_of"] = pd.to_datetime(inst["as_of"])
    return inst


@pytest.fixture
def fills():
    return pd.DataFrame(
        [
            {"trade_id": "T1", "asset": "X", "quantity": 10, "fill_price": 100.0},
            {"trade_id": "T2", "asset": "Q", "quantity": 4, "fill_price": 20.0},  # unknown asset
            {"trade_id": "T3", "asset": "Y", "quantity": 3, "fill_price": 50.0},
        ]
    )


def test_enrich_fills_uses_the_latest_reference_row(fills, instruments):
    out = enrich_fills(fills, instruments)

    assert list(out["trade_id"]) == ["T1", "T2", "T3"], "fills keep their original order"
    assert out.loc[0, "multiplier"] == pytest.approx(2.0), "the 01-02 republish wins, not the 01-01 row"
    assert out.loc[0, "notional"] == pytest.approx(2000.0)
    assert out.loc[2, "notional"] == pytest.approx(750.0)
    assert out.loc[2, "currency"] == "EUR"


def test_enrich_fills_never_multiplies_rows(fills, instruments):
    # X appears twice in the reference table. A join on a repeating key
    # duplicates the left side, and a book that silently doubles a position
    # is worse than one that fails loudly.
    out = enrich_fills(fills, instruments)
    assert len(out) == len(fills) == 3
    assert list(out.index) == [0, 1, 2]


def test_enrich_fills_flags_unknown_assets_instead_of_dropping_them(fills, instruments):
    out = enrich_fills(fills, instruments)
    unknown = out[out["trade_id"] == "T2"].iloc[0]

    assert bool(unknown["unmapped"]) is True
    assert pd.isna(unknown["multiplier"]) and pd.isna(unknown["notional"])
    assert not out.loc[out["trade_id"] != "T2", "unmapped"].any(), "matched fills are not flagged"


def test_enrich_fills_empty_input_still_has_the_columns(instruments):
    empty = pd.DataFrame({"trade_id": [], "asset": [], "quantity": [], "fill_price": []})
    out = enrich_fills(empty, instruments)

    assert len(out) == 0
    for col in ("multiplier", "currency", "notional", "unmapped"):
        assert col in out.columns


# ============================================================
# Problem 5 - Bar The Tape
# ============================================================
@pytest.fixture
def tape():
    # deliberately out of order, and two X trades inside the same 5-min bar
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 09:36:00",
                    "2026-01-01 09:31:00",
                    "2026-01-01 09:38:00",
                    "2026-01-01 09:34:00",
                    "2026-01-01 09:33:00",
                ]
            ),
            "asset": ["X", "X", "Y", "X", "Y"],
            "price": [110.0, 100.0, 52.0, 102.0, 50.0],
            "size": [2.0, 1.0, 6.0, 3.0, 4.0],
        }
    )


def test_bar_summary_buckets_and_weights(tape):
    out = bar_summary(tape, "5min")

    assert list(out.columns) == ["bar", "asset", "vwap", "volume", "share", "prev_vwap"]
    assert list(out.index) == [0, 1, 2, 3]
    assert out["bar"].tolist() == [
        pd.Timestamp("2026-01-01 09:30"),
        pd.Timestamp("2026-01-01 09:30"),
        pd.Timestamp("2026-01-01 09:35"),
        pd.Timestamp("2026-01-01 09:35"),
    ], "bars sit on the freq grid and 09:31 + 09:34 share the 09:30 bar"
    assert out["asset"].tolist() == ["X", "Y", "X", "Y"]
    # X's 09:30 bar: (100*1 + 102*3) / 4 = 101.5, not the plain mean 101.0
    assert out["vwap"].tolist() == pytest.approx([101.5, 50.0, 110.0, 52.0])
    assert out["volume"].tolist() == pytest.approx([4.0, 4.0, 2.0, 6.0])


def test_bar_summary_share_is_the_bars_total_not_the_whole_tape(tape):
    out = bar_summary(tape, "5min")

    assert out["share"].tolist() == pytest.approx([0.5, 0.5, 0.25, 0.75])
    for bar, group in out.groupby("bar"):
        assert group["share"].sum() == pytest.approx(1.0), f"{bar} shares must sum to 1"


def test_bar_summary_prev_vwap_does_not_leak_between_assets(tape):
    out = bar_summary(tape, "5min")

    assert pd.isna(out.loc[0, "prev_vwap"]) and pd.isna(out.loc[1, "prev_vwap"]), (
        "each asset's first bar has no predecessor"
    )
    # Row 2 is X's second bar. Lagging the sorted frame instead of lagging
    # within each asset would hand it row 1's value (Y's 50.0).
    assert out.loc[2, "prev_vwap"] == pytest.approx(101.5)
    assert out.loc[3, "prev_vwap"] == pytest.approx(50.0)


def test_bar_summary_leaves_empty_bars_out():
    tape = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 09:31:00", "2026-01-01 09:52:00"]),
            "asset": ["X", "X"],
            "price": [100.0, 120.0],
            "size": [1.0, 1.0],
        }
    )
    out = bar_summary(tape, "5min")

    assert len(out) == 2, "the four silent bars in between are not rows"
    assert out["share"].tolist() == pytest.approx([1.0, 1.0]), "a lone asset owns its bar"


# ============================================================
# Problem 6 - Track The Book, Realized vs. Unrealized
# ============================================================
def test_track_book_with_realized():
    prices = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"), "X": [108.0, 108.0]})
    filled = pd.DataFrame(
        [
            {"asset": "X", "side": "SELL", "quantity": 10, "fill_price": 100.0},  # maker buys 10 @100
            {"asset": "X", "side": "SELL", "quantity": 5, "fill_price": 103.0},  # maker buys 5 more @103
            {"asset": "X", "side": "BUY", "quantity": 8, "fill_price": 105.0},  # maker sells 8, realizes P&L
        ]
    )
    result = track_book_with_realized(filled, prices)

    assert result["position"]["X"] == pytest.approx(7.0)
    assert result["avg_cost"]["X"] == pytest.approx(101.0)
    assert result["realized_pnl"]["X"] == pytest.approx(32.0)
    assert result["unrealized_pnl"]["X"] == pytest.approx(49.0)
    assert result["total_pnl"] == pytest.approx(81.0)


def test_track_book_with_realized_reconciles():
    # realized + unrealized must always equal the reported total, for any input.
    prices = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"), "X": [95.0, 95.0]})
    filled = pd.DataFrame(
        [
            {"asset": "X", "side": "BUY", "quantity": 20, "fill_price": 90.0},  # maker sells (shorts) 20 @90
            {"asset": "X", "side": "SELL", "quantity": 6, "fill_price": 92.0},  # maker buys back 6, realizes
        ]
    )
    result = track_book_with_realized(filled, prices)
    total_from_parts = sum(result["realized_pnl"].values()) + sum(result["unrealized_pnl"].values())
    assert result["total_pnl"] == pytest.approx(total_from_parts, abs=1e-6)


def test_track_book_with_realized_untraded_asset_stays_flat():
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"),
            "X": [108.0, 108.0],
            "Z": [5.0, 5.0],
        }
    )
    filled = pd.DataFrame([{"asset": "X", "side": "SELL", "quantity": 10, "fill_price": 100.0}])

    result = track_book_with_realized(filled, prices)

    assert result["position"] == pytest.approx({"X": 10.0, "Z": 0.0})
    assert result["avg_cost"]["Z"] == pytest.approx(0.0), "no position, no cost basis"
    assert result["realized_pnl"]["Z"] == pytest.approx(0.0)
    assert result["unrealized_pnl"]["Z"] == pytest.approx(0.0), "a flat book cannot be marked"
    assert result["total_pnl"] == pytest.approx(80.0)


# ============================================================
# Problem 7 - The Daily Scorecard
# ============================================================
@pytest.fixture
def scorecard_fills():
    # X fills twice on day 1 and once on day 2; Y only trades on day 2.
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02 10:00:00",
                    "2026-01-01 09:31:00",
                    "2026-01-01 14:02:00",
                    "2026-01-02 11:00:00",
                ]
            ),
            "asset": ["Y", "X", "X", "X"],
            "quantity": [4, 10, 5, 2],
            "fill_price": [50.0, 100.0, 102.0, 110.0],
        }
    )


def test_daily_scorecard_stats_in_one_aggregation(scorecard_fills):
    stats = daily_scorecard(scorecard_fills)["stats"]

    assert list(stats.columns) == ["trades", "volume", "avg_price", "price_std"]
    assert list(stats.index) == ["X", "Y"], "one row per asset, sorted"
    assert stats.loc["X", "trades"] == 3
    assert stats.loc["X", "volume"] == pytest.approx(17)
    assert stats.loc["X", "avg_price"] == pytest.approx(104.0)
    assert stats.loc["X", "price_std"] == pytest.approx(5.291503, abs=1e-6), "sample std (ddof=1)"


def test_daily_scorecard_single_fill_has_no_price_std(scorecard_fills):
    stats = daily_scorecard(scorecard_fills)["stats"]
    assert stats.loc["Y", "trades"] == 1
    assert pd.isna(stats.loc["Y", "price_std"]), "one print has no spread - NaN, not 0"


def test_daily_scorecard_pivots_to_one_column_per_asset(scorecard_fills):
    daily = daily_scorecard(scorecard_fills)["daily"]

    assert list(daily.columns) == ["X", "Y"]
    assert daily.index.tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")], (
        "dates at midnight, sorted, and only the days that actually traded"
    )
    # X on day 1: 10*100 + 5*102 = 1510 - the two fills of the day are summed,
    # which is why a plain pivot would raise on the repeated (date, asset) pair
    assert daily.loc[pd.Timestamp("2026-01-01"), "X"] == pytest.approx(1510.0)
    assert daily.loc[pd.Timestamp("2026-01-02"), "X"] == pytest.approx(220.0)
    assert daily.loc[pd.Timestamp("2026-01-02"), "Y"] == pytest.approx(200.0)


def test_daily_scorecard_untraded_cell_is_zero_not_nan(scorecard_fills):
    daily = daily_scorecard(scorecard_fills)["daily"]
    assert daily.loc[pd.Timestamp("2026-01-01"), "Y"] == pytest.approx(0.0)
    assert daily.notna().all().all(), "a flat day is 0.0, not a missing observation"


def test_daily_scorecard_cumulative_accumulates_down_the_dates(scorecard_fills):
    out = daily_scorecard(scorecard_fills)
    daily, cumulative = out["daily"], out["cumulative"]

    assert list(cumulative.columns) == list(daily.columns)
    assert cumulative.index.equals(daily.index)
    assert cumulative["X"].tolist() == pytest.approx([1510.0, 1730.0])
    assert cumulative["Y"].tolist() == pytest.approx([0.0, 200.0])
    # the last row of the cumulative table is the column total
    assert cumulative.iloc[-1].tolist() == pytest.approx(daily.sum().tolist())


def test_daily_scorecard_empty_input():
    empty = pd.DataFrame({"timestamp": [], "asset": [], "quantity": [], "fill_price": []})
    out = daily_scorecard(empty)

    assert set(out) == {"stats", "daily", "cumulative"}
    assert all(len(frame) == 0 for frame in out.values())
