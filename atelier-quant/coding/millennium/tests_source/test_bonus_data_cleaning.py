"""Grading suite for the bonus data-cleaning exercise.

Run with:  pytest tests/test_bonus_data_cleaning.py -v

Expected values below were computed once from data_cache/messy_trading_data.csv
via solutions/bonus_data_cleaning_solution.py and hardcoded, the same way
every other day's ground-truth checks work in this program.
"""
from __future__ import annotations

import pandas as pd
import _mini_pytest as pytest


import _cible as mod
load_and_clean = mod.load_and_clean
realized_vol_by_ticker = mod.realized_vol_by_ticker

CSV_PATH = "data_cache/messy_trading_data.csv"


@pytest.fixture
def cleaned():
    return load_and_clean(CSV_PATH)


def test_row_count(cleaned):
    assert len(cleaned) == 144, f"expected 144 surviving rows, got {len(cleaned)}"


def test_tickers_are_clean(cleaned):
    assert set(cleaned["ticker"].unique()) == {"AAPL", "MSFT", "GOOG"}
    assert cleaned["ticker"].str.strip().eq(cleaned["ticker"]).all(), "ticker values still have stray whitespace"


def test_no_bad_volumes_survive(cleaned):
    assert (cleaned["volume"] > 0).all(), "a non-positive volume survived cleaning"
    assert cleaned["volume"].notna().all()


def test_no_duplicates_survive(cleaned):
    assert not cleaned.duplicated().any()


def test_numeric_dtypes(cleaned):
    assert pd.api.types.is_numeric_dtype(cleaned["price"])
    assert pd.api.types.is_numeric_dtype(cleaned["volume"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])


def test_realized_vol_matches_expected(cleaned):
    vol = realized_vol_by_ticker(cleaned)
    expected = {"AAPL": 0.015469, "GOOG": 0.017329, "MSFT": 0.019836}
    for ticker, exp in expected.items():
        assert ticker in vol.index, f"missing ticker {ticker} in realized_vol_by_ticker output"
        assert vol[ticker] == pytest.approx(exp, abs=0.001), (
            f"{ticker}: got {vol[ticker]:.6f}, expected {exp:.6f} - likely a price parsing bug "
            "($ or , not stripped correctly)"
        )
