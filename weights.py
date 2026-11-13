"""The weights model: for any set of stocks, long-only weights summing to 100% with the least ES."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest import REPORTS
from frontier import LEVEL, measure, min_es_weights, scenario_arguments, scenario_label, scenarios
from returns import latest_snapshot, load_prices, log_returns
from simulation import model_arguments, model_options


def model_weights(returns: pd.DataFrame, n: int = 10_000, days: int = 21, seed: int = 0,
                  **scenario_options) -> tuple[pd.Series, pd.DataFrame]:
    """Minimum-ES weights for the stocks in `returns`, and the scenarios they were fitted on.

    The rule is the safest end of the frontier. Moving up the frontier needs expected returns:
    under zero drift those are only the compounding bonus of volatile stocks, and historical
    means are too noisy to act on, so the model uses what the data can pin down, which is how
    the stocks move together. The seed shifts single weights by about a point but not the ES,
    which is flat around its minimum.
    """
    scen = scenarios(returns, n, days, seed, **scenario_options)
    return min_es_weights(scen), scen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--window", type=int, default=252,
                        help="Calibrate on the last this many daily returns, as in walkforward.py")
    parser.add_argument("--symbols", nargs="+", help="Stocks to weight; defaults to all in the snapshot")
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", choices=["zero", "historical"], default="zero")
    model_arguments(parser)
    scenario_arguments(parser)
    parser.add_argument("--out", type=Path, default=REPORTS / "weights.csv")
    args = parser.parse_args()
    try:
        prices = load_prices(args.snapshot or latest_snapshot())
        if args.window < 1:
            raise ValueError("Window must be positive")
        symbols = [s.upper() for s in args.symbols or prices.columns]
        unknown = sorted(set(symbols).difference(prices.columns))
        if unknown or len(set(symbols)) != len(symbols):
            raise ValueError(f"No price history for: {', '.join(unknown)}" if unknown
                             else "Symbols must be unique")
        returns = log_returns(prices[symbols].iloc[-(args.window + 1):])
        weights, scen = model_weights(returns, args.paths, args.days, args.seed, drift=args.drift,
                                      source=args.source, block=args.block, **model_options(args))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    equal = pd.Series(1 / len(symbols), index=symbols)
    table = pd.DataFrame({"model": measure(scen, weights), "equal weight": measure(scen, equal)}).T
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights.rename_axis("symbol").rename("weight").to_csv(args.out)
    print(f"{len(symbols)} stocks; {args.paths:,} scenarios x {args.days} days; "
          f"{scenario_label(args)};\ncalibrated on the last {len(returns)} returns, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}\n")
    for symbol, weight in weights.sort_values(ascending=False).items():
        print(f"  {symbol:<6}{weight:>6.1%}")
    print(f"\nes: mean {args.days}-day loss in the worst {1 - LEVEL:.0%} of scenarios; "
          f"upside: gain beaten only by the best {1 - LEVEL:.0%}.")
    with pd.option_context("display.float_format", "{:.1%}".format):
        print(table.to_string())
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
