"""Simulate correlated GBM price paths for a fixed-share portfolio."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from returns import latest_snapshot, load_prices, log_returns

PORTFOLIO = Path(__file__).parent / "portfolio.csv"
MIN_RETURNS = 60
LAMBDA = 0.94  # RiskMetrics daily decay
NU = 5  # Student-t degrees of freedom: excess kurtosis 6 / (nu - 4) = 6 for each stock


def covariance(returns: pd.DataFrame, method: str = "equal", lam: float = LAMBDA) -> pd.DataFrame:
    """Daily covariance of log returns.

    equal: sample covariance, every day in the window weighs the same.
    ewma: RiskMetrics exponentially weighted covariance around a zero mean. The newest day weighs
    (1 - lam), each older day lam times the one after it; weights are rescaled to sum to one over
    the window. With lam = 0.94 a day's weight halves in about 11 trading days. The result is
    today's covariance and is held constant over the whole simulated horizon (no mean reversion).
    """
    if method == "equal":
        return returns.cov()
    if method != "ewma":
        raise ValueError(f"Unknown covariance {method!r}; use 'equal' or 'ewma'")
    if not 0 < lam < 1:
        raise ValueError("EWMA lambda must be between 0 and 1")
    data = returns.to_numpy(dtype=float)
    weights = lam ** np.arange(len(data))[::-1]  # oldest row first, newest weight 1
    weights /= weights.sum()
    cov = (data * weights[:, None]).T @ data
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def parameters(returns: pd.DataFrame, drift: str = "zero", cov: str = "equal",
               lam: float = LAMBDA) -> tuple[pd.Series, pd.DataFrame]:
    # "zero" sets the mean *log* return to zero: median price is unchanged, but the expected
    # simple return is still positive (exp(sigma^2 T / 2) - 1), larger for volatile stocks.
    # "historical" uses the sample mean log return, which already contains the -sigma^2/2 drag.
    if len(returns) < max(MIN_RETURNS, returns.shape[1] + 1):
        raise ValueError(f"Need at least {MIN_RETURNS} daily returns to calibrate, got {len(returns)}")
    cov = covariance(returns, cov, lam)
    if not np.isfinite(cov.to_numpy()).all():
        raise ValueError("Covariance has non-finite entries")
    if drift == "zero":
        mean = pd.Series(0.0, index=returns.columns)
    elif drift == "historical":
        mean = returns.mean()
    else:
        raise ValueError(f"Unknown drift {drift!r}; use 'zero' or 'historical'")
    return mean, cov


def correlated_shocks(cov: pd.DataFrame, n_paths: int, n_days: int, rng: np.random.Generator,
                      shocks: str = "normal", nu: float = NU) -> np.ndarray:
    """Daily log-return shocks with covariance `cov`, shape (paths, days, assets).

    normal: Cholesky, cov = L @ L.T. If z ~ N(0, I) then L @ z ~ N(0, cov). Volatility lives
    inside L, so shocks are not scaled by sigma again.
    t: multivariate Student-t. Each path-day draws one W ~ chi2(nu) shared by all stocks and
    multiplies the normal shock by sqrt((nu - 2) / W). E[(nu - 2) / W] = 1, so the covariance is
    still `cov` and only the shape changes: more very calm days, more very large ones. Because W is
    shared, large moves arrive together (tail dependence), which correlation alone cannot express.
    Same seed, same z: the t shocks rescale the normal ones, so comparisons share random numbers.
    """
    try:
        lower = np.linalg.cholesky(cov.to_numpy())
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance is not positive definite") from error
    z = rng.standard_normal((n_paths, n_days, len(lower)))
    if shocks == "t":
        if not nu > 2:
            raise ValueError("Student-t needs nu > 2 for a finite variance")
        z *= np.sqrt((nu - 2) / rng.chisquare(nu, (n_paths, n_days, 1)))
    elif shocks != "normal":
        raise ValueError(f"Unknown shocks {shocks!r}; use 'normal' or 't'")
    return z @ lower.T


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
    if table.empty:
        raise ValueError("Portfolio has no positions")
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


def align_positions(symbols: pd.Index, start: pd.Series,
                    shares: pd.Series) -> tuple[pd.Series, pd.Series]:
    # Align by symbol, never by position.
    unknown = shares.index.difference(symbols)
    if len(unknown):
        raise ValueError(f"No price history for: {', '.join(unknown)}")
    shares = shares.reindex(symbols, fill_value=0.0).astype(float)
    if not np.isfinite(shares).all():
        raise ValueError("Shares must be finite")
    if not (shares != 0).any():
        raise ValueError("Portfolio has no positions")
    start = start.reindex(symbols).astype(float)
    if start.isna().any():
        raise ValueError(f"Missing start price for: {', '.join(start.index[start.isna()])}")
    if not np.isfinite(start).all() or (start <= 0).any():
        raise ValueError("Start prices must be finite and positive")
    return start, shares


def simulate(returns: pd.DataFrame, start: pd.Series, shares: pd.Series, n_paths: int = 10_000,
             n_days: int = 21, seed: int = 0, drift: str = "zero", cov: str = "equal",
             lam: float = LAMBDA, shocks: str = "normal", nu: float = NU) -> np.ndarray:
    if n_paths < 1 or n_days < 1:
        raise ValueError("Paths and days must be positive")
    start, shares = align_positions(returns.columns, start, shares)
    mean, matrix = parameters(returns, drift, cov, lam)
    draws = correlated_shocks(matrix, n_paths, n_days, np.random.default_rng(seed), shocks, nu)
    return portfolio_values(price_paths(start, mean, draws), shares)


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD for calibration")
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO)
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", choices=["zero", "historical"], default="zero",
                        help="zero: zero mean log return (not zero expected return); "
                             "historical: sample mean log return")
    model_arguments(parser)


def model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--cov", choices=["equal", "ewma"], default="equal",
                        help="equal: every day in the window weighs the same; "
                             "ewma: recent days weigh more (RiskMetrics)")
    parser.add_argument("--lam", type=float, default=LAMBDA, help="EWMA decay per day")
    parser.add_argument("--shocks", choices=["normal", "t"], default="normal",
                        help="normal: Gaussian daily shocks; t: Student-t with the same covariance "
                             "and fatter tails")
    parser.add_argument("--nu", type=float, default=NU, help="Student-t degrees of freedom (> 2)")


def model_label(args) -> str:
    cov = "equal-weight covariance" if args.cov == "equal" else f"EWMA covariance (lambda {args.lam})"
    return f"{cov}; " + ("normal shocks" if args.shocks == "normal" else f"Student-t shocks (nu {args.nu})")


def model_options(args) -> dict:
    """Keyword arguments for simulate() and backtest() from the parsed model arguments."""
    return {"cov": args.cov, "lam": args.lam, "shocks": args.shocks, "nu": args.nu}


def load_inputs(args) -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """Returns, start prices, shares and a label saying where the start prices came from."""
    prices = load_prices(args.snapshot or latest_snapshot(), args.start)
    portfolio = load_portfolio(args.portfolio)
    given = portfolio.get("price", pd.Series(dtype=float)).notna()
    source = (f"{prices.index[-1].date()} close" if not given.any() else
              "portfolio prices" if given.all() else
              f"portfolio prices, {prices.index[-1].date()} close where blank")
    return (log_returns(prices), start_prices(prices.iloc[-1], portfolio),
            portfolio["shares"].astype(float), source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        returns, start, shares, source = load_inputs(args)
        values = simulate(returns, start, shares, args.paths, args.days, args.seed, args.drift,
                          **model_options(args))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    now, end = values[0, 0], values[:, -1]
    print(f"Start ${now:,.0f} ({source}); {args.paths:,} paths x {args.days} days; "
          f"{args.drift} log drift; {model_label(args)}; calibrated on {len(returns)} returns\n")
    print("Value at horizon end:")
    for q in (1, 5, 25, 50, 75, 95, 99):
        value = np.percentile(end, q)
        print(f"  {q:>2}th percentile  ${value:>9,.0f}  ({value / now - 1:+.1%})")
    print(f"  mean             ${end.mean():>9,.0f}  ({end.mean() / now - 1:+.1%})")


if __name__ == "__main__":
    main()
