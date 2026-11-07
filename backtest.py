"""Rolling VaR backtest: forecast with data known at each date, then compare with the realized loss."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from returns import latest_snapshot, load_prices, log_returns
from risk import LEVELS, losses, var_es
from simulation import MIN_RETURNS, PORTFOLIO, align_positions, load_portfolio, simulate

REPORTS = Path(__file__).parent / "reports"


def backtest(prices: pd.DataFrame, shares: pd.Series, levels=LEVELS, days: int = 21,
             window: int = 252, n_paths: int = 5_000, seed: int = 0, drift: str = "zero",
             step: int = 1) -> pd.DataFrame:
    """One row per forecast date t.

    The model at t sees only prices up to and including t: it is calibrated on the last `window`
    daily returns ending at t and starts from the close at t. The realized loss is
    V_t - V_{t+days} with the same fixed shares and actual closes, so it is never used to fit.
    """
    if window < MIN_RETURNS:
        raise ValueError(f"Window must be at least {MIN_RETURNS} returns")
    if step < 1:
        raise ValueError("Step must be positive")
    _, shares = align_positions(prices.columns, prices.iloc[-1], shares)
    rows = []
    for i in range(window, len(prices) - days, step):
        known = prices.iloc[i - window:i + 1]  # window + 1 closes -> window returns, ending at t
        start = known.iloc[-1]
        values = simulate(log_returns(known), start, shares, n_paths, days, seed + i, drift)
        simulated = losses(values)
        value_now = float(start @ shares)
        realized = value_now - float(prices.iloc[i + days] @ shares)
        if not np.isfinite(realized):
            raise ValueError(f"Realized loss at {prices.index[i].date()} is not finite")
        # days and window travel with each row so charts never have to guess the settings.
        row = {"date": prices.index[i], "end": prices.index[i + days], "days": days,
               "window": window, "value": value_now, "loss": realized}
        for level in levels:
            var, es = var_es(simulated, level)
            row[f"var_{level:.0%}"] = var
            row[f"es_{level:.0%}"] = es  # the model's own forecast of the average loss in a breach
            row[f"breach_{level:.0%}"] = realized > var
        rows.append(row)
    if not rows:
        raise ValueError(f"Need more than {window + days} price rows for one forecast")
    return pd.DataFrame(rows).set_index("date")


def kupiec(n: int, breaches: int, level: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test: LR statistic and p-value (chi-squared, 1 dof).

    Assumes independent outcomes, so only feed it non-overlapping windows.
    """
    if n < 1 or not 0 <= breaches <= n:
        raise ValueError("Need 0 <= breaches <= n and n >= 1")
    p, rate = 1 - level, breaches / n

    def log_lik(q):  # 0 * log(0) is taken as 0
        return ((n - breaches) * math.log(1 - q) if breaches < n else 0.0) + \
               (breaches * math.log(q) if breaches else 0.0)

    lr = max(0.0, -2 * (log_lik(p) - log_lik(rate)))
    return lr, math.erfc(math.sqrt(lr / 2))  # survival function of chi2(1)


def non_overlapping(results: pd.DataFrame, days: int, step: int) -> pd.DataFrame:
    # Forecasts whose horizons do not share any day, starting from the first one.
    if days % step:
        raise ValueError("Days must be a multiple of step to pick non-overlapping windows")
    return results.iloc[::days // step]


def summarize(results: pd.DataFrame, levels, days: int, step: int) -> pd.DataFrame:
    independent = non_overlapping(results, days, step)
    rows = []
    for level in levels:
        breach = f"breach_{level:.0%}"
        n, x = len(independent), int(independent[breach].sum())
        lr, p_value = kupiec(n, x, level)
        rows.append({"level": level, "expected": 1 - level,
                     "all_n": len(results), "all_rate": results[breach].mean(),
                     "indep_n": n, "indep_breaches": x, "indep_rate": x / n,
                     "kupiec_lr": lr, "p_value": p_value})
    return pd.DataFrame(rows).set_index("level")


def breach_episodes(results: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Group breaches into episodes: runs of breaching forecasts whose loss windows overlap.

    Overlapping daily forecasts share most of their horizon, so one bad stretch shows up as many
    breaches. An episode ends when the next breach starts after every earlier window has ended.
    Losses, VaR and ES are shares of the portfolio value at each forecast date.
    """
    var, es, breach = f"var_{level:.0%}", f"es_{level:.0%}", f"breach_{level:.0%}"
    hits = results[results[breach].astype(bool)]
    if hits.empty:
        return pd.DataFrame(columns=["first", "last", "through", "breaches", "worst", "loss",
                                     "var", "es", "excess", "loss_to_var"])
    hits = hits.assign(loss_pct=hits["loss"] / hits["value"], var_pct=hits[var] / hits["value"],
                       es_pct=hits[es] / hits["value"])
    ends = pd.to_datetime(hits["end"])
    # A window starting on the day an earlier one ends shares no returns with it: new episode.
    episode = (hits.index >= ends.cummax().shift()).cumsum()
    rows = []
    for _, group in hits.groupby(np.asarray(episode)):
        worst = group["loss_pct"].idxmax()
        rows.append({"first": group.index[0], "last": group.index[-1],
                     "through": pd.to_datetime(group["end"]).max(), "breaches": len(group),
                     "worst": worst, "loss": group.at[worst, "loss_pct"],
                     "var": group.at[worst, "var_pct"], "es": group.at[worst, "es_pct"],
                     "excess": group.at[worst, "loss_pct"] - group.at[worst, "var_pct"],
                     "loss_to_var": group.at[worst, "loss_pct"] / group.at[worst, "var_pct"]})
    return pd.DataFrame(rows)


def exceedance_summary(results: pd.DataFrame, level: float = 0.95) -> dict[str, float]:
    """How far realized losses went past VaR, and whether the model's ES anticipated it."""
    var, es, breach = f"var_{level:.0%}", f"es_{level:.0%}", f"breach_{level:.0%}"
    hits = results[results[breach].astype(bool)]
    if hits.empty:
        return {"breaches": 0}
    loss, value = hits["loss"], hits["value"]
    return {"breaches": len(hits), "episodes": len(breach_episodes(results, level)),
            "mean_excess": ((loss - hits[var]) / value).mean(),
            "max_excess": ((loss - hits[var]) / value).max(),
            "mean_loss_to_var": (loss / hits[var]).mean(),
            "mean_loss": (loss / value).mean(), "mean_es": (hits[es] / value).mean()}


def print_breaches(results: pd.DataFrame, level: float = 0.95):
    summary = exceedance_summary(results, level)
    print(f"\n{level:.0%} VaR breaches: {summary['breaches']} of {len(results)} forecasts", end="")
    if not summary["breaches"]:
        print()
        return
    print(f", in {summary['episodes']} episodes of overlapping windows")
    print(f"  Past VaR by {summary['mean_excess']:.1%} of value on average "
          f"(worst {summary['max_excess']:.1%}); loss / VaR averages {summary['mean_loss_to_var']:.2f}")
    print(f"  Mean loss in a breach {summary['mean_loss']:.1%} vs. the model's ES "
          f"{summary['mean_es']:.1%} on those same dates")
    print(f"\n  {'first':<12}{'last':<12}{'window ends':<13}{'n':>3}  {'worst start':<13}"
          f"{'loss':>7}{'VaR':>7}{'ES':>7}{'excess':>8}{'loss/VaR':>9}")
    for row in breach_episodes(results, level).itertuples():
        print(f"  {row.first.date()!s:<12}{row.last.date()!s:<12}{row.through.date()!s:<13}"
              f"{row.breaches:>3}  {row.worst.date()!s:<13}{row.loss:>7.1%}{row.var:>7.1%}"
              f"{row.es:>7.1%}{row.excess:>8.1%}{row.loss_to_var:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Snapshot folder; defaults to the latest READY one")
    parser.add_argument("--start", help="First price date YYYY-MM-DD")
    parser.add_argument("--portfolio", type=Path, default=PORTFOLIO)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--window", type=int, default=252, help="Calibration returns per forecast")
    parser.add_argument("--paths", type=int, default=5_000)
    parser.add_argument("--step", type=int, default=1, help="Trading days between forecasts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", choices=["zero", "historical"], default="zero")
    parser.add_argument("--out", type=Path, default=REPORTS / "backtest.csv")
    args = parser.parse_args()
    try:
        if args.days < 1:
            raise ValueError("Days must be positive")
        snapshot = args.snapshot or latest_snapshot()
        prices = load_prices(snapshot, args.start)
        # Historical closes are the start prices here; the portfolio's price column is ignored.
        shares = load_portfolio(args.portfolio)["shares"].astype(float)
        results = backtest(prices, shares, LEVELS, args.days, args.window, args.paths,
                           args.seed, args.drift, args.step)
        table = summarize(results, LEVELS, args.days, args.step)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out)
    print(f"{snapshot.name}: {len(results)} forecasts of {args.days}-day loss, "
          f"{results.index[0].date()} to {results.index[-1].date()}; "
          f"{args.window}-return rolling window; {args.paths:,} paths; {args.drift} log drift\n")
    print(f"  {'level':<7}{'expected':<10}{'all (overlapping)':<20}"
          f"{'non-overlapping':<20}{'Kupiec LR':<11}p-value")
    for level, row in table.iterrows():
        print(f"  {level:<7.0%}{row.expected:<10.1%}"
              f"{f'{row.all_rate:.1%} of {row.all_n:.0f}':<20}"
              f"{f'{row.indep_breaches:.0f}/{row.indep_n:.0f} = {row.indep_rate:.1%}':<20}"
              f"{row.kupiec_lr:<11.2f}{row.p_value:.3f}")
    print("\nOverlapping windows share days, so their breaches cluster: use the rate as a description,"
          "\nand the non-overlapping column for the test. A low p-value rejects the breach rate.")
    print_breaches(results, 0.95)
    print(f"Saved per-date results to {args.out}")


if __name__ == "__main__":
    main()
