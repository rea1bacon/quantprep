"""Bonus material: a deliberately messy trades export.

Every other dataset in this program is clean synthetic data - great for
testing quant logic, useless for practicing the unglamorous pandas work a
real "Dataset Assessment" almost certainly includes: inconsistent column
names, mixed number formats, bad rows that need to be dropped, duplicates.
This module builds exactly that, from an underlying clean series so the
"correct" answer after cleaning is still well-defined and checkable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market_data import generate_gbm_path

TICKERS_AND_START_PRICES = [("AAPL", 180.0), ("MSFT", 380.0), ("GOOG", 1450.0)]


def generate_clean_trades(seed: int = 7, n_rows_per_ticker: int = 60) -> pd.DataFrame:
    """The ground-truth data, before anything gets messy. One row every 5
    minutes per ticker, starting at market open."""
    start = pd.Timestamp("2026-01-05 09:30:00")
    rng = np.random.default_rng(seed)

    rows = []
    for i, (ticker, s0) in enumerate(TICKERS_AND_START_PRICES):
        prices = generate_gbm_path(n_rows_per_ticker, s0=s0, mu=0.05, sigma=0.25, dt=1 / 252, seed=seed + i)
        for j, price in enumerate(prices):
            rows.append(
                {
                    "timestamp": start + pd.Timedelta(minutes=5 * j),
                    "ticker": ticker,
                    "price": round(float(price), 2),
                    "volume": int(rng.integers(50, 500)),
                }
            )
    return pd.DataFrame(rows)


def messify(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Corrupt formatting (never the underlying numbers of rows that survive
    cleaning) so there's a well-defined 'correct' cleaned result."""
    rng = np.random.default_rng(seed)
    messy = df.copy()

    def mess_ticker(t: str) -> str:
        r = rng.random()
        if r < 0.3:
            return t.lower()
        if r < 0.5:
            return f" {t} "
        return t

    def mess_price(p: float) -> str:
        r = rng.random()
        if r < 0.05:
            return ""
        if r < 0.10:
            return "N/A"
        if p >= 1000:
            return f"${p:,.2f}"
        if r < 0.4:
            return f"${p:.2f}"
        return f"{p:.2f}"

    def mess_volume(v: int):
        r = rng.random()
        if r < 0.05:
            return ""
        if r < 0.10:
            return -abs(v)
        if r < 0.30:
            return f"{v:,}"
        return v

    messy["ticker"] = messy["ticker"].map(mess_ticker)
    messy["price"] = messy["price"].map(mess_price)
    messy["volume"] = messy["volume"].map(mess_volume)

    # duplicate a handful of rows exactly, verbatim
    dup_idx = rng.choice(messy.index, size=max(1, len(messy) // 25), replace=False)
    messy = pd.concat([messy, messy.loc[dup_idx]], ignore_index=True)

    # real exports aren't sorted, and rarely have clean headers
    messy = messy.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    messy = messy.rename(
        columns={"timestamp": " Timestamp", "ticker": "Ticker ", "price": "Price", "volume": "Volume"}
    )
    return messy


def write_messy_csv(
    path: str = "data_cache/messy_trading_data.csv", seed_clean: int = 7, seed_mess: int = 11
) -> None:
    clean = generate_clean_trades(seed=seed_clean)
    messy = messify(clean, seed=seed_mess)
    messy.to_csv(path, index=False)


if __name__ == "__main__":
    write_messy_csv()
    print("wrote data_cache/messy_trading_data.csv")
