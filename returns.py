"""Turn a validated price snapshot into daily log returns and summary statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
TRADING_DAYS = 252


def latest_snapshot(data_dir: Path = DATA) -> Path:
    # Snapshot folder names embed a UTC timestamp, so name order is time order.
    ready = [p.parent for p in sorted(data_dir.glob("yahoo_*/manifest.json"))
             if json.loads(p.read_text(encoding="utf-8")).get("status") == "READY"]
    if not ready:
        raise FileNotFoundError(f"No READY snapshot under {data_dir}")
    return ready[-1]


def load_prices(snapshot: Path, start: str | None = None) -> pd.DataFrame:
    prices = pd.read_csv(snapshot / "prices.csv", index_col="date", parse_dates=True)
    if start:
        prices = prices.loc[pd.Timestamp(start):]
    if len(prices) < 2:
        raise ValueError("Need at least two price rows")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # r_t = ln(P_t / P_{t-1}); daily log returns add up to the multi-day log return.
    return np.log(prices).diff().iloc[1:]


def summary(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        # Annualized mean log return: not CAGR, nor the expected simple return.
        "ann_log_mean": returns.mean() * TRADING_DAYS,
        "ann_vol": returns.std() * np.sqrt(TRADING_DAYS),
        "worst_day": np.expm1(returns.min()),
        "best_day": np.expm1(returns.max()),
        "skew": returns.skew(),
        "excess_kurtosis": returns.kurt(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD; defaults to the full common history")
    args = parser.parse_args()
    snapshot = args.snapshot or latest_snapshot()
    returns = log_returns(load_prices(snapshot, args.start))
    print(f"{snapshot.name}: {len(returns)} daily returns, "
          f"{returns.index.min().date()} to {returns.index.max().date()}\n")
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 120):
        print(summary(returns).sort_values("ann_vol", ascending=False), "\n")
        print("Correlation of daily log returns:")
        print(returns.corr())


if __name__ == "__main__":
    main()
