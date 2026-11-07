"""Risk metrics from simulated portfolio values: VaR, expected shortfall, drawdown, risk shares."""

from __future__ import annotations

import argparse
from statistics import NormalDist

import numpy as np
import pandas as pd

from simulation import add_arguments, align_positions, load_inputs, simulate

LEVELS = (0.95, 0.99)


def losses(values: np.ndarray, day: int = -1) -> np.ndarray:
    # L = V_0 - V_day: a positive number is a loss.
    return values[:, 0] - values[:, day]


def var_es(loss: np.ndarray, level: float) -> tuple[float, float]:
    # Empirical tail: the worst k = ceil(N * (1 - level)) outcomes.
    # VaR is the mildest loss in that tail, ES is the tail mean, so ES >= VaR by construction.
    if not 0 < level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if len(loss) == 0:
        raise ValueError("No losses to measure")
    k = int(np.ceil(round(len(loss) * (1 - level), 9)))
    tail = np.sort(loss)[-k:]
    return float(tail[0]), float(tail.mean())


def max_drawdown(values: np.ndarray) -> np.ndarray:
    # Largest fall from a running peak along each path; day 0 counts as a peak.
    peaks = np.maximum.accumulate(values, axis=1)
    return (1 - values / peaks).max(axis=1)


def variance_contribution(returns: pd.DataFrame, start: pd.Series, shares: pd.Series) -> pd.DataFrame:
    # Euler split of *daily* portfolio variance: w_i (Cov w)_i / w'Cov w, summing to 1.
    # Not a contribution to the 21-day simulated VaR, which is a different (non-linear) quantity.
    start, shares = align_positions(returns.columns, start, shares)
    exposure = start * shares
    cov = returns.cov()
    marginal = cov @ exposure
    return pd.DataFrame({"weight": exposure / exposure.sum(),
                         "variance_share": exposure * marginal / (exposure @ marginal)})


def parametric_var(returns: pd.DataFrame, start: pd.Series, shares: pd.Series,
                   level: float, days: int) -> float:
    # Approximate reference only: treats the portfolio return as normal and linear (delta-normal,
    # zero mean). The simulation compounds lognormal prices, so the two can differ at long horizons.
    start, shares = align_positions(returns.columns, start, shares)
    exposure = start * shares
    sigma = np.sqrt(exposure @ returns.cov() @ exposure * days)
    return float(NormalDist().inv_cdf(level) * sigma)


def gbm_var_es(start: float, mean: float, vol: float, days: int, level: float) -> tuple[float, float]:
    # Exact VaR and ES of L = P_0 - P_T for one GBM stock with daily log mean m and daily vol s:
    # P_T = P_0 exp(mT + s sqrt(T) Z), so the loss tail is Z <= z, with z the (1 - level) quantile.
    a, b = mean * days, vol * np.sqrt(days)
    z = NormalDist().inv_cdf(1 - level)
    var = start * (1 - np.exp(a + b * z))
    # E[exp(a + bZ) | Z <= z] = exp(a + b^2 / 2) * Phi(z - b) / Phi(z)
    tail_mean = np.exp(a + b * b / 2) * NormalDist().cdf(z - b) / (1 - level)
    return float(var), float(start * (1 - tail_mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        returns, start, shares, source = load_inputs(args)
        values = simulate(returns, start, shares, args.paths, args.days, args.seed, args.drift)
        contribution = variance_contribution(returns, start, shares)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    now = values[0, 0]
    pct = lambda x: f"${x:>7,.0f} ({x / now:5.1%})"
    print(f"Start ${now:,.0f} ({source}); {args.paths:,} paths x {args.days} days; "
          f"{args.drift} log drift; calibrated on {len(returns)} returns\n")
    print("Losses (L = start value - value at horizon; positive is a loss):")
    print(f"  {'horizon':<8}{'level':<7}{'VaR':<19}{'ES (CVaR)':<19}parametric VaR (approx.)")
    for day in sorted({1, args.days}):
        for level in LEVELS:
            var, es = var_es(losses(values, day), level)
            ref = parametric_var(returns, start, shares, level, day)
            print(f"  {f'{day}d':<8}{level:<7.0%}{pct(var):<19}{pct(es):<19}{pct(ref)}")
    drawdown = max_drawdown(values)
    print(f"\nMax drawdown within {args.days} days: median {np.median(drawdown):.1%}, "
          f"95th pct {np.percentile(drawdown, 95):.1%}, 99th pct {np.percentile(drawdown, 99):.1%}")
    print("\nShare of daily portfolio variance (not of the 21-day VaR):")
    with pd.option_context("display.float_format", "{:.1%}".format):
        print(contribution.sort_values("variance_share", ascending=False).to_string())


if __name__ == "__main__":
    main()
