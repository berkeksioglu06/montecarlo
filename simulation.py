"""Simulate correlated GBM price paths for a fixed-share portfolio."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from returns import latest_snapshot, load_prices, log_returns

PORTFOLIO = Path(__file__).parent / "portfolio.csv"
MIN_RETURNS = 60


def parameters(returns: pd.DataFrame, drift: str = "zero") -> tuple[pd.Series, pd.DataFrame]:
    # "zero" sets the mean *log* return to zero: median price is unchanged, but the expected
    # simple return is still positive (exp(sigma^2 T / 2) - 1), larger for volatile stocks.
    # "historical" uses the sample mean log return, which already contains the -sigma^2/2 drag.
    if len(returns) < max(MIN_RETURNS, returns.shape[1] + 1):
        raise ValueError(f"Need at least {MIN_RETURNS} daily returns to calibrate, got {len(returns)}")
    cov = returns.cov()
    if not np.isfinite(cov.to_numpy()).all():
        raise ValueError("Covariance has non-finite entries")
    if drift == "zero":
        mean = pd.Series(0.0, index=returns.columns)
    elif drift == "historical":
        mean = returns.mean()
    else:
        raise ValueError(f"Unknown drift {drift!r}; use 'zero' or 'historical'")
    return mean, cov


def correlated_shocks(cov: pd.DataFrame, n_paths: int, n_days: int,
                      rng: np.random.Generator) -> np.ndarray:
    # Cholesky: cov = L @ L.T. If z ~ N(0, I) then L @ z ~ N(0, cov).
    # Volatility lives inside L, so shocks are not scaled by sigma again.
    try:
        lower = np.linalg.cholesky(cov.to_numpy())
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance is not positive definite") from error
    z = rng.standard_normal((n_paths, n_days, len(lower)))
    return z @ lower.T  # shape: (paths, days, assets)


def price_paths(start: pd.Series, mean: pd.Series, shocks: np.ndarray) -> np.ndarray:
    # P_t = P_0 * exp(sum of daily log returns up to t); day 0 is the starting price.
    log_paths = np.cumsum(mean.to_numpy() + shocks, axis=1)
    log_paths = np.concatenate([np.zeros_like(log_paths[:, :1]), log_paths], axis=1)
    return start.to_numpy() * np.exp(log_paths)  # shape: (paths, days + 1, assets)


def portfolio_values(paths: np.ndarray, shares: pd.Series) -> np.ndarray:
    # Fixed share counts, no trading during the horizon. Shares must be ordered like paths.
    return paths @ shares.to_numpy(dtype=float)  # shape: (paths, days + 1)


def load_portfolio(path: Path) -> pd.DataFrame:
    # Columns: symbol, shares, and optionally price (start price; blank uses the last close).
    table = pd.read_csv(path)
    if not {"symbol", "shares"}.issubset(table.columns):
        raise ValueError("Portfolio needs symbol and shares columns")
    if table["symbol"].duplicated().any():
        raise ValueError("Duplicate symbols in portfolio")
    table = table.set_index("symbol")
    for column in table.columns.intersection(["shares", "price"]):
        values = table[column].dropna() if column == "price" else table[column]
        values = values.astype(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Portfolio {column} must be finite and positive")
    return table


def start_prices(last_close: pd.Series, portfolio: pd.DataFrame) -> pd.Series:
    start = last_close.astype(float).copy()
    if "price" in portfolio:
        given = portfolio["price"].dropna().astype(float)
        start.loc[given.index.intersection(start.index)] = given
    return start


def simulate(returns: pd.DataFrame, start: pd.Series, shares: pd.Series, n_paths: int = 10_000,
             n_days: int = 21, seed: int = 0, drift: str = "zero") -> np.ndarray:
    # Align by symbol, never by position.
    symbols = returns.columns
    unknown = shares.index.difference(symbols)
    if len(unknown):
        raise ValueError(f"No price history for: {', '.join(unknown)}")
    shares = shares.reindex(symbols, fill_value=0.0)
    start = start.reindex(symbols)
    if start.isna().any():
        raise ValueError(f"Missing start price for: {', '.join(start.index[start.isna()])}")
    mean, cov = parameters(returns, drift)
    shocks = correlated_shocks(cov, n_paths, n_days, np.random.default_rng(seed))
    return portfolio_values(price_paths(start, mean, shocks), shares)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD for calibration")
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", choices=["zero", "historical"], default="zero",
                        help="zero: zero mean log return (not zero expected return); "
                             "historical: sample mean log return")
    args = parser.parse_args()
    try:
        prices = load_prices(args.snapshot or latest_snapshot(), args.start)
        returns = log_returns(prices)
        portfolio = load_portfolio(args.portfolio)
        start = start_prices(prices.iloc[-1], portfolio)
        values = simulate(returns, start, portfolio["shares"].astype(float), args.paths,
                          args.days, args.seed, args.drift)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    given = portfolio.get("price", pd.Series(dtype=float)).notna()
    source = (f"{prices.index[-1].date()} close" if not given.any() else
              "portfolio prices" if given.all() else
              f"portfolio prices, {prices.index[-1].date()} close where blank")
    now, end = values[0, 0], values[:, -1]
    print(f"Start ${now:,.0f} ({source}); {args.paths:,} paths x {args.days} days; "
          f"{args.drift} log drift; calibrated on {len(returns)} returns\n")
    print("Value at horizon end:")
    for q in (1, 5, 25, 50, 75, 95, 99):
        value = np.percentile(end, q)
        print(f"  {q:>2}th percentile  ${value:>9,.0f}  ({value / now - 1:+.1%})")
    print(f"  mean             ${end.mean():>9,.0f}  ({end.mean() / now - 1:+.1%})")


if __name__ == "__main__":
    main()
