"""Grading suite for the numpy vectorization drills.

Run with:  python run.py test numpy      (or: pytest tests/test_numpy_drills.py -v)

Expected values were computed once from solutions/numpy_drills_solution.py
on the fixtures below and hardcoded, the same way every other suite in this
program works.
"""
from __future__ import annotations

import numpy as np
import _mini_pytest as pytest


import _cible as mod
rolling_zscore = mod.rolling_zscore
asof_indices = mod.asof_indices
vwap_by_label = mod.vwap_by_label
max_drawdown = mod.max_drawdown
factor_betas = mod.factor_betas
top_k_indices = mod.top_k_indices
winsorize_zscore = mod.winsorize_zscore
ffill_2d = mod.ffill_2d
book_exposure = mod.book_exposure


# ============================================================
# Problem 1 - Rolling Z-Score
# ============================================================
def test_rolling_zscore_basic():
    prices = np.array([100.0, 101.0, 102.0, 103.0, 110.0])
    out = rolling_zscore(prices, 3)

    assert out.shape == prices.shape
    assert np.isnan(out[:2]).all(), "the first window-1 values have no full window -> NaN"
    # windows [100,101,102] and [101,102,103] are perfectly linear ramps:
    # the last point sits exactly +1 sample-sigma above the window mean.
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(1.0)
    assert out[4] == pytest.approx(1.147079, abs=1e-6)


def test_rolling_zscore_uses_sample_std_not_population():
    prices = np.array([1.0, 2.0, 3.0])
    out = rolling_zscore(prices, 3)
    # mean 2, sample std (ddof=1) = 1.0 -> z = 1.0.
    # A population std (ddof=0) would be 0.8165 and give 1.2247 instead.
    assert out[2] == pytest.approx(1.0), "use the sample std (ddof=1)"


def test_rolling_zscore_flat_window_is_zero_not_inf():
    prices = np.array([50.0, 50.0, 50.0, 50.0])
    out = rolling_zscore(prices, 3)
    assert np.isfinite(out[2:]).all(), "a zero-std window must not produce inf/nan"
    assert out[2:].tolist() == pytest.approx([0.0, 0.0])


def test_rolling_zscore_window_longer_than_series():
    out = rolling_zscore(np.array([1.0, 2.0]), 5)
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_rolling_zscore_window_of_one():
    # A window of one has no spread at all, so every point sits exactly on its
    # own mean. The sample std of a single value is undefined (ddof=1 divides
    # by zero): the zero-std rule must catch it instead of returning nan.
    out = rolling_zscore(np.array([5.0, 7.0, 9.0]), 1)
    assert np.isfinite(out).all()
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0])



# ============================================================
# Problem 2 - Vectorized As-Of Lookup
# ============================================================
def _ts(*strings):
    return np.array([np.datetime64(s) for s in strings], dtype="datetime64[ns]")


def test_asof_indices_backward_and_exact_match():
    tape = _ts("2026-01-01T09:30", "2026-01-01T09:35", "2026-01-01T09:40")
    queries = _ts("2026-01-01T09:37", "2026-01-01T09:35", "2026-01-01T09:41")

    out = asof_indices(tape, queries, max_staleness_minutes=10)

    assert np.issubdtype(out.dtype, np.integer), "return an integer array"
    assert out.tolist() == [1, 1, 2], "an exact hit uses that bar, never the next one"


def test_asof_indices_before_first_print_is_minus_one():
    tape = _ts("2026-01-01T09:30", "2026-01-01T09:35")
    queries = _ts("2026-01-01T09:00", "2026-01-01T09:29:59")
    assert asof_indices(tape, queries, 60).tolist() == [-1, -1]


def test_asof_indices_staleness_boundary_is_inclusive():
    tape = _ts("2026-01-01T09:30")
    queries = _ts("2026-01-01T09:34:59", "2026-01-01T09:35", "2026-01-01T09:35:01")
    out = asof_indices(tape, queries, max_staleness_minutes=5)
    assert out.tolist() == [0, 0, -1], "exactly max_staleness old is still acceptable"


def test_asof_indices_unsorted_queries_and_empty():
    tape = _ts("2026-01-01T09:30", "2026-01-01T09:40")
    queries = _ts("2026-01-01T09:45", "2026-01-01T09:31")
    assert asof_indices(tape, queries, 60).tolist() == [1, 0], "results follow query order"
    assert asof_indices(tape, _ts(), 60).tolist() == []


# ============================================================
# Problem 3 - Aggregate By Label
# ============================================================
def test_vwap_by_label_weights_by_size_and_sorts_labels():
    prices = np.array([100.0, 102.0, 50.0])
    sizes = np.array([1.0, 3.0, 10.0])
    labels = np.array(["X", "X", "Y"])

    uniques, vwap = vwap_by_label(prices, sizes, labels)

    assert uniques.tolist() == ["X", "Y"]
    assert vwap.tolist() == pytest.approx([101.5, 50.0]), "size-weighted, not the plain mean (101.0)"


def test_vwap_by_label_handles_unsorted_repeated_labels():
    prices = np.array([50.0, 100.0, 60.0, 102.0])
    sizes = np.array([10.0, 1.0, 10.0, 3.0])
    labels = np.array(["Y", "X", "Y", "X"])

    uniques, vwap = vwap_by_label(prices, sizes, labels)

    assert uniques.tolist() == ["X", "Y"], "the distinct labels come back sorted"
    assert vwap.tolist() == pytest.approx([101.5, 55.0])


def test_vwap_by_label_zero_size_is_nan():
    uniques, vwap = vwap_by_label(np.array([10.0, 20.0]), np.array([0.0, 0.0]), np.array(["Z", "Z"]))
    assert uniques.tolist() == ["Z"]
    assert np.isnan(vwap).all(), "no volume means no VWAP, not a divide-by-zero"


def test_vwap_by_label_empty_input():
    uniques, vwap = vwap_by_label(np.array([]), np.array([]), np.array([], dtype="<U4"))
    assert uniques.size == 0 and vwap.size == 0


# ============================================================
# Problem 4 - Max Drawdown
# ============================================================
def test_max_drawdown_basic():
    equity = np.array([100.0, 120.0, 90.0, 110.0, 105.0])
    out = max_drawdown(equity)

    assert out["max_drawdown"] == pytest.approx(-0.25, abs=1e-9), "90 vs a peak of 120"
    assert out["peak_idx"] == 1
    assert out["trough_idx"] == 2


def test_max_drawdown_monotonic_curve():
    out = max_drawdown(np.array([100.0, 101.0, 102.0]))
    assert out["max_drawdown"] == 0.0
    assert out["peak_idx"] == 0 and out["trough_idx"] == 0


def test_max_drawdown_reports_first_of_tied_troughs():
    # Two identical -20% drawdowns; the first one wins.
    equity = np.array([100.0, 80.0, 100.0, 80.0])
    out = max_drawdown(equity)
    assert out["max_drawdown"] == pytest.approx(-0.2)
    assert out["trough_idx"] == 1
    assert out["peak_idx"] == 0


def test_max_drawdown_empty():
    assert max_drawdown(np.array([])) == {"max_drawdown": 0.0, "peak_idx": -1, "trough_idx": -1}


# ============================================================
# Problem 5 - Multi-Factor Betas
# ============================================================
def test_factor_betas_recovers_exact_relationship():
    rng = np.random.default_rng(7)
    factors = rng.normal(size=(200, 2))
    y = 0.001 + 1.5 * factors[:, 0] - 0.4 * factors[:, 1]

    out = factor_betas(y, factors)

    assert out["alpha"] == pytest.approx(0.001, abs=1e-6)
    assert out["betas"].shape == (2,), "betas excludes the intercept"
    assert out["betas"] == pytest.approx([1.5, -0.4], abs=1e-6)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_factor_betas_are_multivariate_not_univariate():
    # The two factors are correlated, so running two separate univariate
    # regressions gives visibly different (wrong) betas.
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=300)
    f2 = 0.8 * f1 + 0.6 * rng.normal(size=300)
    factors = np.column_stack([f1, f2])
    y = 2.0 * f1 + 1.0 * f2 + 0.01 * rng.normal(size=300)

    out = factor_betas(y, factors)

    assert out["betas"] == pytest.approx([2.0, 1.0], abs=2e-3)
    assert 0.99 < out["r_squared"] <= 1.0


def test_factor_betas_constant_asset_returns_has_zero_r2():
    factors = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.5, 0.5, 0.5, 0.5])

    out = factor_betas(y, factors)

    assert out["alpha"] == pytest.approx(0.5, abs=1e-6)
    assert out["betas"] == pytest.approx([0.0], abs=1e-6)
    assert out["r_squared"] == 0.0, "SS_tot == 0 must not divide by zero"


def test_factor_betas_survives_rank_deficient_factors():
    # The second factor is exactly twice the first, so X.T @ X is singular and
    # an explicit inverse would blow up. lstsq returns the minimum-norm
    # solution instead, which still reproduces the data exactly.
    f1 = np.arange(6.0)
    factors = np.column_stack([f1, 2 * f1])
    y = 3 * f1 + 1.0

    out = factor_betas(y, factors)

    assert np.isfinite(out["betas"]).all(), "a singular factor matrix must not produce nan/inf"
    assert out["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-6)
    # the two collinear betas must still reconstruct the 3x slope
    assert out["betas"][0] + 2 * out["betas"][1] == pytest.approx(3.0, abs=1e-6)



# ============================================================
# Problem 6 - Top-K Per Row
# ============================================================
def test_top_k_indices_orders_descending():
    scores = np.array(
        [
            [0.1, 0.9, 0.5, 0.3],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    out = top_k_indices(scores, 2)

    assert out.shape == (2, 2)
    assert np.issubdtype(out.dtype, np.integer)
    assert out[0].tolist() == [1, 2]
    assert out[1].tolist() == [0, 1]


def test_top_k_indices_breaks_ties_toward_lower_column():
    scores = np.array([[5.0, 5.0, 1.0]])
    assert top_k_indices(scores, 2)[0].tolist() == [0, 1]


def test_top_k_indices_ranks_nan_last():
    scores = np.array([[np.nan, 2.0, 7.0, np.nan]])
    out = top_k_indices(scores, 3)
    assert out[0, :2].tolist() == [2, 1]
    assert out[0, 2] in (0, 3), "the leftover slot falls back to a NaN column"


def test_top_k_indices_degenerate_k():
    scores = np.array([[3.0, 1.0, 2.0]])
    assert top_k_indices(scores, 0).shape == (1, 0)
    assert top_k_indices(scores, 10)[0].tolist() == [0, 2, 1], "k >= n_cols is a full argsort"


# ============================================================
# Problem 7 - Winsorize Then Standardize
# ============================================================
def test_winsorize_zscore_clips_before_standardizing():
    # 1..19 plus one absurd print. The 10th/90th percentiles are 2.9 and
    # 18.1, so both ends get pulled in and the 1000 lands on the cap.
    signal = np.concatenate([np.arange(1.0, 20.0), [1000.0]])
    out = winsorize_zscore(signal, 10, 90)

    assert out.shape == signal.shape
    assert out[0] == pytest.approx(-1.3863, abs=1e-4)
    assert out[1] == pytest.approx(-1.3863, abs=1e-4), "both ends are clipped, not just the top"
    assert out[2] == pytest.approx(-1.368059, abs=1e-6)
    assert out[-1] == pytest.approx(out[-2]), (
        "the 1000 outlier must be clipped down onto the cap, not merely rescaled"
    )
    assert abs(out[-1]) < 2.0, "clipping first keeps every z-score inside a sane range"


def test_winsorize_zscore_keeps_nans_and_ignores_them():
    signal = np.array([1.0, np.nan, 2.0, 3.0])
    out = winsorize_zscore(signal, 0, 100)

    assert np.isnan(out[1]), "NaNs stay NaN"
    assert np.isfinite(out[[0, 2, 3]]).all()
    assert out[[0, 2, 3]] == pytest.approx([-1.0, 0.0, 1.0])


def test_winsorize_zscore_flat_after_clipping():
    signal = np.array([1.0, 5.0, 5.0, 5.0, 9.0])
    out = winsorize_zscore(signal, 25, 75)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_winsorize_zscore_all_nan():
    signal = np.array([np.nan, np.nan])
    out = winsorize_zscore(signal, 5, 95)
    assert out.shape == (2,)
    assert np.isnan(out).all()


# ============================================================
# Problem 8 - Forward-Fill A 2-D Tape
# ============================================================
def test_ffill_2d_fills_down_each_column_independently():
    tape = np.array(
        [
            [100.0, np.nan],
            [np.nan, 50.0],
            [np.nan, np.nan],
            [103.0, 52.0],
        ]
    )
    out = ffill_2d(tape)

    assert out[:, 0].tolist() == pytest.approx([100.0, 100.0, 100.0, 103.0])
    assert np.isnan(out[0, 1]), "a leading NaN has nothing above it - it stays NaN"
    assert out[1:, 1].tolist() == pytest.approx([50.0, 50.0, 52.0])


def test_ffill_2d_never_backfills():
    tape = np.array([[np.nan], [np.nan], [7.0]])
    out = ffill_2d(tape)
    assert np.isnan(out[:2, 0]).all(), "a value below must never travel upward"
    assert out[2, 0] == 7.0


def test_ffill_2d_does_not_mutate_input():
    tape = np.array([[1.0, np.nan], [np.nan, 2.0]])
    original = tape.copy()
    ffill_2d(tape)
    assert np.array_equal(tape, original, equal_nan=True), "return a new array"


def test_ffill_2d_all_nan_column():
    tape = np.array([[np.nan, 1.0], [np.nan, np.nan]])
    out = ffill_2d(tape)
    assert np.isnan(out[:, 0]).all()
    assert out[:, 1].tolist() == pytest.approx([1.0, 1.0])


# ============================================================
# Problem 9 - Value The Whole Book At Once
# ============================================================
def test_book_exposure_applies_the_fx_rate_per_asset():
    positions = np.array([[10.0, -5.0], [4.0, 4.0]])
    prices = np.array([[100.0, 50.0], [100.0, 100.0]])
    fx_rates = np.array([1.0, 2.0])  # the second asset is quoted in another currency

    out = book_exposure(positions, prices, fx_rates)

    # day 0: 10*100*1 = 1000, -5*50*2 = -500 -> net 500, gross 1500
    assert out["value"].tolist() == pytest.approx([500.0, 1200.0])
    assert out["gross"].tolist() == pytest.approx([1500.0, 1200.0])
    assert out["value"].shape == (2,) and out["gross"].shape == (2,)


def test_book_exposure_weights_sum_to_one_per_day():
    positions = np.array([[10.0, -5.0], [4.0, 4.0]])
    prices = np.array([[100.0, 50.0], [100.0, 100.0]])
    fx_rates = np.array([1.0, 2.0])

    out = book_exposure(positions, prices, fx_rates)

    assert out["weights"].shape == (2, 2)
    assert out["weights"][0].tolist() == pytest.approx([2.0, -1.0]), "a short leg gives a negative weight"
    assert out["weights"].sum(axis=1) == pytest.approx([1.0, 1.0])


def test_book_exposure_hedged_day_has_zero_weights():
    # long 1 and short 1 of the same price: net value is exactly 0, so the
    # division has to be guarded rather than producing inf/nan.
    out = book_exposure(np.array([[1.0, -1.0]]), np.array([[100.0, 100.0]]), np.array([1.0, 1.0]))

    assert out["value"].tolist() == pytest.approx([0.0])
    assert np.isfinite(out["weights"]).all()
    # pytest.approx rejects nested lists; compare the array itself
    assert out["weights"] == pytest.approx(np.zeros((1, 2)))
    assert out["gross"].tolist() == pytest.approx([200.0]), "gross still sees both legs"


def test_book_exposure_does_not_tile_the_fx_rates():
    # A single day and many assets: fx_rates lines up on the trailing axis
    # and must broadcast on its own.
    positions = np.ones((1, 4))
    prices = np.full((1, 4), 10.0)
    fx_rates = np.array([1.0, 2.0, 3.0, 4.0])

    out = book_exposure(positions, prices, fx_rates)

    assert out["value"].tolist() == pytest.approx([100.0])
    assert out["gross"].tolist() == pytest.approx([100.0])
