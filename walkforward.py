"""Out-of-sample test: re-optimize weights with data known at each date, hold, compare outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import REPORTS
from frontier import (LEVEL, against_current, check_weights, current_weights, measure,
                      min_es_weights, scenario_arguments, scenario_label)
from returns import latest_snapshot, load_prices, log_returns
from risk import var_es
from simulation import MIN_RETURNS, PORTFOLIO, load_portfolio, model_arguments, model_options
from weights import model_weights

STRATEGIES = ("current", "equal weight", "min ES at same mean", "min ES")


def walk_forward(prices: pd.DataFrame, current: pd.Series, days: int = 21, window: int = 252,
                 n: int = 5_000, seed: int = 0, step: int | None = None,
                 **scenario_options) -> pd.DataFrame:
    """One row per rebalance date t: the realized `days`-day return of each strategy.

    At t the weights model sees only the last `window` daily returns ending at t. Weights are set at
    the close of t and held without trading; the outcome uses the actual closes at t and t + days,
    which are never used to fit. `current` is today's weights applied at every date, so it is a
    fixed mix and not the portfolio that was actually held back then.
    "min ES" is the weights model; the columns after the strategies are its weights at each date.
    """
    if window < MIN_RETURNS:
        raise ValueError(f"Window must be at least {MIN_RETURNS} returns")
    step = days if step is None else step
    if days < 1 or step < 1:
        raise ValueError("Days and step must be positive")
    current = check_weights(current, prices.columns)
    equal = pd.Series(1 / len(current), index=current.index)
    rows = []
    for i in range(window, len(prices) - days, step):
        safest, scen = model_weights(log_returns(prices.iloc[i - window:i + 1]), n, days, seed + i,
                                     **scenario_options)
        # Same model mean as the current mix if reachable, else the safest mix already beats it.
        target = measure(scen, current)["mean"]
        matched = safest if measure(scen, safest)["mean"] >= target else min_es_weights(scen, target)
        realized = prices.iloc[i + days] / prices.iloc[i] - 1
        outcome = {name: float(weights @ realized)
                   for name, weights in zip(STRATEGIES, (current, equal, matched, safest))}
        rows.append({"date": prices.index[i], "end": prices.index[i + days], **outcome, **safest})
    if not rows:
        raise ValueError(f"Need more than {window + days} price rows for one rebalance")
    return pd.DataFrame(rows).set_index("date")


def summarize(results: pd.DataFrame, level: float = LEVEL) -> pd.DataFrame:
    """Realized results per strategy, measured exactly like the model's forecasts in frontier.py.

    es is the mean loss in the worst (1 - level) of periods and upside the gain beaten only by
    the best (1 - level): with about 48 periods that is three observations each, so read them as
    evidence and not as proof. es_cut and upside_kept compare with the current weights.
    """
    rows = {}
    for name in STRATEGIES:
        gain = results[name].to_numpy()
        rows[name] = {"mean": gain.mean(), "worst": gain.min(), "es": var_es(-gain, level)[1],
                      "upside": var_es(gain, level)[0]}
    return against_current(pd.DataFrame(rows).T)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD")
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--window", type=int, default=252, help="Calibration returns per rebalance")
    parser.add_argument("--paths", type=int, default=5_000, help="Scenarios per rebalance")
    parser.add_argument("--step", type=int, help="Trading days between rebalances; defaults to "
                                                 "--days, so holding periods do not overlap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", choices=["zero", "historical"], default="zero")
    model_arguments(parser)
    scenario_arguments(parser)
    parser.add_argument("--out", type=Path, default=REPORTS / "walkforward.csv")
    args = parser.parse_args()
    try:
        prices = load_prices(args.snapshot or latest_snapshot(), args.start)
        shares = load_portfolio(args.portfolio)["shares"].astype(float)
        prices = prices[prices.columns.intersection(shares.index)]  # only reweight what is held
        # Today's weights: share counts valued at the snapshot's last close.
        current = current_weights(prices.columns, prices.iloc[-1], shares)
        results = walk_forward(prices, current, args.days, args.window, args.paths, args.seed,
                               args.step, drift=args.drift, source=args.source, block=args.block,
                               **model_options(args))
        table = summarize(results)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.assign(days=args.days, window=args.window, scenarios=scenario_label(args)).to_csv(args.out)
    print(f"{len(results)} rebalances, {results.index[0].date()} to {results.index[-1].date()}; "
          f"weights held {args.days} days; {args.window}-return rolling window; "
          f"{args.paths:,} scenarios;\n{scenario_label(args)}\n")
    print(f"Realized {args.days}-day returns. es: mean loss in the worst {1 - LEVEL:.0%} of periods; "
          f"upside: gain beaten only by the best {1 - LEVEL:.0%}.")
    with pd.option_context("display.float_format", "{:.1%}".format, "display.width", 160):
        print(table.to_string(), "\n")
        print("Average 'min ES' weights and how far they ranged:")
        weights = results[prices.columns]
        print(pd.DataFrame({"current": current, "mean": weights.mean(), "min": weights.min(),
                            "max": weights.max()}).to_string())
        moved = weights.diff().abs().sum(axis=1).iloc[1:].mean() / 2
        print(f"From one rebalance to the next they move {moved:.1%} of the portfolio on average.")
    print(f"\nSaved per-date results to {args.out}")


if __name__ == "__main__":
    main()
