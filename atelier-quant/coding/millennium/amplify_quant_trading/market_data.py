"""Synthetic but statistically honest market data.

Nothing here is toy data dressed up to look real - every generator plants a
ground-truth parameter (a volatility regime, a true beta, a true cointegrating
relationship) that your code is expected to recover. That's what makes the
exercises checkable: "does your estimate converge to the number we planted"
is a much harder bar than "does your function run."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gbm_path(
    n_steps: int,
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a geometric Brownian motion price path with n_steps points (including s0)."""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    rng = np.random.default_rng(seed)
    increments = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps - 1)
    log_path = np.concatenate([[np.log(s0)], np.log(s0) + np.cumsum(increments)])
    return np.exp(log_path)


def generate_trading_day(
    n_ticks: int = 23_400,
    s0: float = 100.0,
    calm_vol: float = 0.15,
    stress_vol: float = 0.55,
    stress_start_frac: float = 0.55,
    stress_len_frac: float = 0.12,
    seed: int | None = None,
) -> pd.DataFrame:
    """One trading day of tick data with a calm regime and a volatility-spike regime.

    n_ticks=23_400 is one tick per second across a 6.5h trading day. The
    'true_vol' column is the ground-truth annualized instantaneous vol used to
    generate that tick - your realized-vol *estimate* (Day 1) will lag it
    because it's computed from a rolling window, which is itself part of the
    lesson: estimation lag is real risk, not a bug to eliminate.
    """
    if not (0.0 <= stress_start_frac < 1.0) or not (0.0 < stress_len_frac <= 1.0):
        raise ValueError("stress window fractions must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252 * n_ticks)

    true_vol = np.full(n_ticks, calm_vol)
    stress_start = int(n_ticks * stress_start_frac)
    stress_end = min(n_ticks, stress_start + int(n_ticks * stress_len_frac))
    true_vol[stress_start:stress_end] = stress_vol

    shocks = rng.normal(-0.5 * true_vol**2 * dt, true_vol * np.sqrt(dt))
    log_price = np.log(s0) + np.cumsum(shocks)
    log_price = np.concatenate([[np.log(s0)], log_price[:-1]])
    price = np.exp(log_price)

    df = pd.DataFrame({"tick": np.arange(n_ticks), "price": price, "true_vol": true_vol})
    df.attrs["stress_window"] = (stress_start, stress_end)
    df.attrs["dt"] = dt  # pass this to simulate_market_making so realized_vol annualizes correctly
    return df


def generate_stock_etf_pair(
    n_days: int = 252,
    s0_etf: float = 100.0,
    s0_stock: float = 60.0,
    beta: float = 0.72,
    etf_vol: float = 0.18,
    idio_vol: float = 0.25,
    mu_etf: float = 0.06,
    seed: int | None = None,
) -> pd.DataFrame:
    """A stock and an ETF whose returns satisfy r_stock = beta * r_etf + idiosyncratic noise.

    True beta is stashed in df.attrs["true_beta"] so your Day 3 estimator has
    a ground truth to be scored against - a beta estimate off by 0.3 quietly
    turns a hedge into a leveraged bet, which is the entire point of the
    exercise.
    """
    if n_days < 30:
        raise ValueError("n_days should be >= 30 for a stable beta estimate")
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252

    etf_ret = rng.normal((mu_etf - 0.5 * etf_vol**2) * dt, etf_vol * np.sqrt(dt), n_days)
    idio_ret = rng.normal(-0.5 * idio_vol**2 * dt, idio_vol * np.sqrt(dt), n_days)
    stock_ret = beta * etf_ret + idio_ret

    etf_price = s0_etf * np.exp(np.cumsum(np.concatenate([[0.0], etf_ret[:-1]])))
    stock_price = s0_stock * np.exp(np.cumsum(np.concatenate([[0.0], stock_ret[:-1]])))

    dates = pd.bdate_range("2026-01-02", periods=n_days)
    df = pd.DataFrame({"date": dates, "etf": etf_price, "stock": stock_price}).set_index("date")
    df.attrs["true_beta"] = beta
    return df


def generate_etf_basket(
    n_days: int = 252,
    n_components: int = 4,
    s0_etf: float = 100.0,
    market_vol: float = 0.19,
    idio_vol: float = 0.22,
    mispricing_theta: float = 4.0,
    mispricing_eta: float = 0.004,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """An ETF plus its component basket, with a genuinely mean-reverting mispricing term.

    Construction: etf_price = synthetic_price * (1 + x_t), where x_t follows an
    Ornstein-Uhlenbeck process (dx = -theta*x*dt + eta*dW) that mean-reverts to
    zero. This is not decoration - it means ETF and synthetic are truly
    cointegrated, so an Engle-Granger test on your Day 4 output should find a
    significant relationship, and the OU half-life ln(2)/theta is a real,
    checkable number your code can estimate from data alone.

    Returns (dataframe, weights) where weights maps component ticker -> weight.
    """
    if n_components < 2:
        raise ValueError("need at least 2 components for a basket")
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252

    tickers = [f"C{i+1}" for i in range(n_components)]
    weights = dict(zip(tickers, rng.dirichlet(np.ones(n_components))))
    s0_components = rng.uniform(20.0, 80.0, n_components)

    market_ret = rng.normal(0.02 * dt, market_vol * np.sqrt(dt), n_days)
    comp_prices = np.zeros((n_days, n_components))
    for i in range(n_components):
        idio_ret = rng.normal(-0.5 * idio_vol**2 * dt, idio_vol * np.sqrt(dt), n_days)
        comp_ret = 0.9 * market_ret + idio_ret
        comp_prices[:, i] = s0_components[i] * np.exp(np.cumsum(np.concatenate([[0.0], comp_ret[:-1]])))

    w = np.array([weights[t] for t in tickers])
    synthetic = comp_prices @ w

    mispricing = np.zeros(n_days)
    for t in range(1, n_days):
        mispricing[t] = (
            mispricing[t - 1]
            - mispricing_theta * mispricing[t - 1] * dt
            + mispricing_eta * rng.normal() * np.sqrt(dt)
        )
    etf_price = synthetic * (1 + mispricing)

    dates = pd.bdate_range("2026-01-02", periods=n_days)
    data = {"ETF": etf_price, "synthetic_true": synthetic}
    for i, t in enumerate(tickers):
        data[t] = comp_prices[:, i]
    df = pd.DataFrame(data, index=dates)
    df.attrs["true_mispricing"] = mispricing
    df.attrs["ou_half_life_days"] = np.log(2) / mispricing_theta * 252
    return df, weights
