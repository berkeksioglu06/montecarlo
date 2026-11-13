"""Reweight the same stocks: least 21-day expected shortfall for each level of expected return."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from backtest import REPORTS
from risk import var_es
from simulation import (LAMBDA, NU, add_arguments, align_positions, correlated_shocks, load_inputs,
                        model_label, model_options, parameters)

LEVEL = 0.95
BLOCK = 5  # bootstrap block length in trading days


def scenarios(returns: pd.DataFrame, n: int = 10_000, days: int = 21, seed: int = 0,
              drift: str = "zero", source: str = "gbm", block: int = BLOCK, cov: str = "equal",
              lam: float = LAMBDA, shocks: str = "normal", nu: float = NU) -> pd.DataFrame:
    """Simple `days`-day return of every stock, one row per scenario.

    Every set of weights is judged on these same rows, so two portfolios differ only by their
    weights and never by simulation luck.

    gbm: the simulation model. Symmetric log returns, so any asymmetry comes from compounding.
    bootstrap: stitches together blocks of `block` consecutive real days. A whole day is taken
    for all stocks at once, so joint crashes, skew and (within a block) volatility clustering
    stay as they were in the data.
    drift "zero" centers each stock's mean daily log return at zero in both sources; "historical"
    keeps the sample mean. That gives a 2 x 2 comparison: drift assumption x return shape.
    """
    if n < 1 or days < 1:
        raise ValueError("Scenarios and days must be positive")
    rng = np.random.default_rng(seed)
    if source == "gbm":
        mean, matrix = parameters(returns, drift, cov, lam)
        daily = mean.to_numpy() + correlated_shocks(matrix, n, days, rng, shocks, nu)
    elif source == "bootstrap":
        mean, _ = parameters(returns, drift)  # same length and drift checks as the simulation
        data = returns.to_numpy(dtype=float) - returns.mean().to_numpy() + mean.to_numpy()
        if not 1 <= block <= len(data):
            raise ValueError("Block must be between 1 and the number of returns")
        starts = rng.integers(0, len(data) - block + 1, (n, -(-days // block)))
        rows = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :days]
        daily = data[rows]  # shape: (scenarios, days, assets)
    else:
        raise ValueError(f"Unknown source {source!r}; use 'gbm' or 'bootstrap'")
    return pd.DataFrame(np.expm1(daily.sum(axis=1)), columns=returns.columns)


def check_weights(weights: pd.Series, symbols: pd.Index) -> pd.Series:
    weights = weights.reindex(symbols).astype(float)
    if weights.isna().any() or (weights < -1e-9).any() or abs(weights.sum() - 1) > 1e-6:
        raise ValueError("Weights must cover every stock, be non-negative and sum to 1")
    return weights


def measure(scen: pd.DataFrame, weights: pd.Series, level: float = LEVEL) -> dict[str, float]:
    """Mean, bad case and good case of a buy-and-hold portfolio, as shares of today's value.

    With no trading inside the horizon the portfolio return is exactly sum_i w_i R_i.
    es: mean loss in the worst (1 - level) of scenarios. upside: the gain that only the best
    (1 - level) of scenarios beat, i.e. the mirror image of VaR on the winning side.
    """
    gain = scen.to_numpy() @ check_weights(weights, scen.columns).to_numpy()
    return {"mean": float(gain.mean()), "es": var_es(-gain, level)[1],
            "upside": var_es(gain, level)[0]}


def min_es_weights(scen: pd.DataFrame, target: float | None = None,
                   level: float = LEVEL) -> pd.Series:
    """Long-only weights summing to 1 with the smallest expected shortfall.

    Rockafellar-Uryasev: ES(w) = min over a of  a + mean(max(loss_s(w) - a, 0)) / (1 - level),
    and the minimizing a is the VaR. With u_s >= loss_s - a, u_s >= 0 and loss_s = -R_s . w this
    is a linear program in (w, a, u), so the optimum is global. `target`, if given, also requires
    a mean return of at least `target`; the mean is the only reward that keeps the problem linear.
    """
    if not 0 < level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    r = scen.to_numpy(dtype=float)
    s, n = r.shape
    # Variables: n weights, then a, then s shortfalls u.
    cost = np.concatenate([np.zeros(n), [1.0], np.full(s, 1 / ((1 - level) * s))])
    upper = sparse.hstack([-r, -np.ones((s, 1)), -sparse.eye(s)], format="csr")  # -R w - a - u <= 0
    limit = np.zeros(s)
    if target is not None:
        reward = np.concatenate([-r.mean(axis=0), np.zeros(s + 1)])  # -mean . w <= -target
        upper, limit = sparse.vstack([upper, reward[None, :]], format="csr"), np.append(limit, -target)
    budget = np.concatenate([np.ones(n), np.zeros(s + 1)])[None, :]  # weights sum to 1
    bounds = [(0, None)] * n + [(None, None)] + [(0, None)] * s
    result = linprog(cost, A_ub=upper, b_ub=limit, A_eq=budget, b_eq=[1.0], bounds=bounds,
                     method="highs")
    if not result.success:
        raise ValueError(f"No portfolio reaches a mean return of {target:.2%}" if target is not None
                         else f"Optimizer failed: {result.message}")
    weights = np.clip(result.x[:n], 0, None)
    return pd.Series(weights / weights.sum(), index=scen.columns)


def frontier(scen: pd.DataFrame, points: int = 15, level: float = LEVEL) -> pd.DataFrame:
    """Least-ES portfolio for each mean-return target, from the safest mix to the best single stock."""
    if points < 2:
        raise ValueError("Need at least two frontier points")
    safest = min_es_weights(scen, level=level)
    best = (scen.mean() == scen.mean().max()).astype(float)  # all in the highest-mean stock
    low, high = measure(scen, safest, level)["mean"], measure(scen, best / best.sum(), level)["mean"]
    mixes = [safest, *(min_es_weights(scen, target, level)
                       for target in np.linspace(low, high, points)[1:-1]), best / best.sum()]
    return pd.DataFrame([{**measure(scen, weights, level), **weights} for weights in mixes])


def random_mixes(scen: pd.DataFrame, n: int = 1_000, seed: int = 0,
                 level: float = LEVEL) -> pd.DataFrame:
    """Random long-only weightings: the background that shows what the frontier is the edge of."""
    draws = np.random.default_rng(seed).dirichlet(np.full(scen.shape[1], 0.5), n)
    return pd.DataFrame([measure(scen, pd.Series(w, index=scen.columns), level) for w in draws])


def current_weights(symbols: pd.Index, start: pd.Series, shares: pd.Series) -> pd.Series:
    start, shares = align_positions(symbols, start, shares)
    value = start * shares
    return value / value.sum()


def compare(scen: pd.DataFrame, current: pd.Series, level: float = LEVEL) -> pd.DataFrame:
    """The portfolio as it is and three reference mixes, all measured on the same scenarios."""
    equal = pd.Series(1 / scen.shape[1], index=scen.columns)
    mixes = {"current": current, "equal weight": equal,
             "min ES at same mean": min_es_weights(scen, measure(scen, current, level)["mean"], level),
             "min ES": min_es_weights(scen, level=level)}
    return pd.DataFrame({name: {**measure(scen, w, level), **w} for name, w in mixes.items()}).T


def against_current(table: pd.DataFrame) -> pd.DataFrame:
    """Add the share of the current portfolio's ES that is removed and of its upside that is kept."""
    now = table.loc["current"]
    return table.assign(es_cut=1 - table["es"] / now["es"],
                        upside_kept=table["upside"] / now["upside"])


def scenario_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--source", choices=["gbm", "bootstrap"], default="gbm",
                        help="gbm: the simulation model; bootstrap: blocks of real days")
    parser.add_argument("--block", type=int, default=BLOCK, help="Bootstrap block length in days")


def scenario_label(args) -> str:
    source = model_label(args) if args.source == "gbm" else f"bootstrap of {args.block}-day blocks"
    return f"{args.drift} log drift; {source}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    scenario_arguments(parser)
    parser.add_argument("--points", type=int, default=15, help="Portfolios along the frontier")
    parser.add_argument("--out", type=Path, default=REPORTS / "frontier.csv")
    args = parser.parse_args()
    try:
        returns, start, shares, _ = load_inputs(args)
        returns = returns[returns.columns.intersection(shares.index)]  # only reweight what is held
        current = current_weights(returns.columns, start, shares)
        scen = scenarios(returns, args.paths, args.days, args.seed, args.drift, args.source,
                         args.block, **model_options(args))
        table, curve = compare(scen, current), frontier(scen, args.points)
        cloud = random_mixes(scen, seed=args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    saved = pd.concat([table.rename_axis("portfolio").reset_index(),
                       curve.assign(portfolio="frontier"), cloud.assign(portfolio="random")])
    saved.assign(days=args.days, scenarios=scenario_label(args)).to_csv(args.out, index=False)
    metrics = ["mean", "es", "upside", "es_cut", "upside_kept"]
    print(f"{args.paths:,} shared scenarios x {args.days} days; {scenario_label(args)}; "
          f"calibrated on {len(returns)} returns\n")
    print(f"mean: expected {args.days}-day return. es: mean loss in the worst {1 - LEVEL:.0%} of "
          f"scenarios.\nupside: gain beaten only by the best {1 - LEVEL:.0%}. es_cut and "
          "upside_kept compare with the current portfolio.\n")
    with pd.option_context("display.float_format", "{:.1%}".format, "display.width", 160):
        print(against_current(table)[metrics].to_string(), "\n")
        print("Weights:")
        print(table[returns.columns].to_string(), "\n")
        print("Frontier: least ES for each mean-return target")
        print(curve.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
