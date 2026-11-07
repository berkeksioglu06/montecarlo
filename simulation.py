"""Simulate correlated GBM price paths for a fixed-share portfolio."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from returns import latest_snapshot, load_prices, log_returns

PORTFOLIO = Path(__file__).parent / "portfolio.csv"


def parameters(returns: pd.DataFrame, zero_mean: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    # Daily mean *log* return already contains the -sigma^2/2 drag; do not subtract it again.
    # Zero mean is the default: historical means are noisy forecasts and small next to volatility.
    mean = pd.Series(0.0, index=returns.columns) if zero_mean else returns.mean()
    return mean, returns.cov()


def correlated_shocks(cov: pd.DataFrame, n_paths: int, n_days: int,
                      rng: np.random.Generator) -> np.ndarray:
    # Cholesky: cov = L @ L.T. If z ~ N(0, I) then L @ z ~ N(0, cov).
    # Volatility lives inside L, so shocks are not scaled by sigma again.
    lower = np.linalg.cholesky(cov.to_numpy())
    z = rng.standard_normal((n_paths, n_days, len(lower)))
    return z @ lower.T  # shape: (paths, days, assets)


def price_paths(start: pd.Series, mean: pd.Series, shocks: np.ndarray) -> np.ndarray:
    # P_t = P_0 * exp(sum of daily log returns up to t); day 0 is the starting price.
    log_paths = np.cumsum(mean.to_numpy() + shocks, axis=1)
    log_paths = np.concatenate([np.zeros_like(log_paths[:, :1]), log_paths], axis=1)
    return start.to_numpy() * np.exp(log_paths)  # shape: (paths, days + 1, assets)


def portfolio_values(paths: np.ndarray, shares: pd.Series) -> np.ndarray:
    # Fixed share counts, no trading during the horizon.
    return paths @ shares.to_numpy(dtype=float)  # shape: (paths, days + 1)


def load_portfolio(path: Path, symbols: pd.Index) -> pd.Series:
    table = pd.read_csv(path)
    if table["symbol"].duplicated().any():
        raise ValueError("Duplicate symbols in portfolio")
    shares = table.set_index("symbol")["shares"].astype(float)
    if not np.isfinite(shares).all() or (shares <= 0).any():
        raise ValueError("Share counts must be finite and positive")
    unknown = shares.index.difference(symbols)
    if len(unknown):
        raise ValueError(f"No price history for: {', '.join(unknown)}")
    # Align to the return columns so shares and prices line up by position.
    return shares.reindex(symbols, fill_value=0.0)


def simulate(returns: pd.DataFrame, start: pd.Series, shares: pd.Series, n_paths: int = 10_000,
             n_days: int = 21, seed: int = 0, zero_mean: bool = True) -> np.ndarray:
    mean, cov = parameters(returns, zero_mean)
    shocks = correlated_shocks(cov, n_paths, n_days, np.random.default_rng(seed))
    return portfolio_values(price_paths(start[returns.columns], mean, shocks), shares)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD for calibration")
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--historical-mean", action="store_true",
                        help="Use the historical mean log return instead of zero drift")
    args = parser.parse_args()
    prices = load_prices(args.snapshot or latest_snapshot(), args.start)
    returns = log_returns(prices)
    shares = load_portfolio(args.portfolio, returns.columns)
    values = simulate(returns, prices.iloc[-1], shares, args.paths, args.days, args.seed,
                      zero_mean=not args.historical_mean)
    now, end = values[0, 0], values[:, -1]
    print(f"Start {prices.index[-1].date()}: ${now:,.0f}; {args.paths:,} paths x {args.days} days\n")
    print("Value at horizon end:")
    for q in (1, 5, 25, 50, 75, 95, 99):
        value = np.percentile(end, q)
        print(f"  {q:>2}th percentile  ${value:>9,.0f}  ({value / now - 1:+.1%})")
    print(f"  mean             ${end.mean():>9,.0f}  ({end.mean() / now - 1:+.1%})")


if __name__ == "__main__":
    main()
