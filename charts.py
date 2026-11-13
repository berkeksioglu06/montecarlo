"""Charts: VaR backtest, simulated 21-day outcomes, return correlations, reweighting frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from backtest import REPORTS  # noqa: E402
from risk import losses, var_es  # noqa: E402
from simulation import add_arguments, load_inputs, model_options, simulate  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
VAR_95, VAR_99 = "#6da7ec", "#184f95"  # one blue ramp: lighter = milder level
BREACH = "#d03b3b"
BLUES = "Blues"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
}


def backtest_settings(results: pd.DataFrame) -> tuple[int, int]:
    """Horizon and calibration window recorded by backtest.py, never taken from chart arguments."""
    missing = {"days", "window"}.difference(results.columns)
    if missing:
        raise ValueError(f"Backtest results lack {', '.join(sorted(missing))}; rerun backtest.py")
    settings = []
    for column in ("days", "window"):
        values = results[column].unique()
        if len(values) != 1:
            raise ValueError(f"Backtest results mix several {column} settings: {sorted(values)}")
        settings.append(int(values[0]))
    return settings[0], settings[1]


def backtest_labels(results: pd.DataFrame) -> dict[str, str]:
    days, window = backtest_settings(results)
    return {"title": f"Rolling {days}-day VaR vs. realized loss",
            "subtitle": f"Each date: model fit only on the last {window} daily returns up to that "
                        f"date; loss measured over the next {days} trading days",
            "loss": f"Realized {days}-day loss"}


def backtest_chart(results: pd.DataFrame, path: Path) -> Path:
    # Everything as a share of the portfolio value at the forecast date, so the scale is comparable
    # across years even though fixed shares make the dollar value drift.
    labels = backtest_labels(results)
    loss = results["loss"] / results["value"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.plot(loss.index, loss, color=INK_2, linewidth=1, label=labels["loss"])
        for level, color in (("95%", VAR_95), ("99%", VAR_99)):
            var = results[f"var_{level}"] / results["value"]
            breach = results[f"breach_{level}"].astype(bool)
            ax.plot(var.index, var, color=color, linewidth=2,
                    label=f"{level} VaR ({breach.mean():.1%} breached, expected {100 - int(level[:2])}%)")
        hits = results["breach_95%"].astype(bool)
        ax.scatter(loss.index[hits], loss[hits], s=22, marker="o", color=BREACH, zorder=3,
                   edgecolors=SURFACE, linewidths=1, label="Loss above 95% VaR")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Share of portfolio value (positive = loss)")
        ax.set_title(labels["title"], pad=24)
        ax.text(0, 1.02, labels["subtitle"], transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def paths_chart(values: np.ndarray, path: Path, days: int = 21, n_show: int = 200,
                seed: int = 0) -> Path:
    # A sample of paths shows what single scenarios look like; the bands are computed from
    # *all* paths and show where most of them end up. One color: a path's color carries no meaning.
    now = values[0, 0]
    x = np.arange(values.shape[1])
    sample = np.random.default_rng(seed).choice(len(values), min(n_show, len(values)), replace=False)
    p = np.percentile(values, [1, 5, 25, 50, 75, 95], axis=0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(x, values[sample].T, color=VAR_95, linewidth=0.6, alpha=0.35)
        ax.fill_between(x, p[1], p[5], color=VAR_99, alpha=0.12, linewidth=0, label="5th to 95th percentile")
        ax.fill_between(x, p[2], p[4], color=VAR_99, alpha=0.22, linewidth=0, label="25th to 75th percentile")
        ax.plot(x, p[3], color=VAR_99, linewidth=2, label="Median")
        ax.plot(x, p[0], color=INK, linewidth=1.5, linestyle="--", label="1st percentile")
        ax.axhline(now, color=INK_2, linewidth=0.8)
        for q, label in ((p[0, -1], "1%"), (p[1, -1], "5%"), (p[3, -1], "50%"), (p[5, -1], "95%")):
            ax.annotate(f"{label}  ${q:,.0f} ({q / now - 1:+.0%})", (x[-1], q), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, x[-1])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(min(p[0].min(), values[sample].min()) * 0.97, values[sample].max() * 1.03)
        ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
        ax.set_xlabel("Trading days from today")
        ax.set_ylabel("Portfolio value")
        ax.set_title(f"Simulated portfolio value: {len(sample)} of {len(values):,} paths", pad=24)
        ax.text(0, 1.02, f"Start ${now:,.0f}; bands and labels use all {len(values):,} paths",
                transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def distribution_chart(values: np.ndarray, path: Path, days: int = 21) -> Path:
    now = values[0, 0]
    change = -losses(values) / now  # simple return of the portfolio; negative = loss
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.hist(change, bins=120, color=VAR_95, edgecolor=SURFACE, linewidth=0.3)
        ax.grid(axis="x", visible=False)
        for level, style in ((0.95, "-"), (0.99, "--")):
            var, es = var_es(losses(values), level)
            ax.axvline(-var / now, color=VAR_99, linewidth=2, linestyle=style,
                       label=f"{level:.0%} VaR: loss {var / now:.1%}  (ES {es / now:.1%})")
        ax.legend(loc="upper right", fontsize=9, title="Worst-tail thresholds", title_fontsize=9)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(f"Portfolio change after {days} trading days")
        ax.set_ylabel("Simulated paths")
        ax.set_title(f"Simulated {days}-day outcomes ({len(change):,} paths)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def correlation_chart(returns: pd.DataFrame, path: Path) -> Path:
    corr = returns.corr()
    n = len(corr)
    masked = corr.where(np.tril(np.ones((n, n), dtype=bool)))  # each pair once
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(masked, cmap=BLUES, vmin=0, vmax=1)
        ax.set_xticks(range(n), corr.columns)
        ax.set_yticks(range(n), corr.index)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(n):
            for j in range(i + 1):
                value = corr.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color=SURFACE if value > 0.6 else INK)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
        ax.set_title(f"Correlation of daily log returns\n{returns.index[0].date()} to "
                     f"{returns.index[-1].date()}", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def frontier_chart(table: pd.DataFrame, path: Path) -> Path:
    """Bad case against good case for the portfolios saved by frontier.py."""
    days, scenarios = int(table["days"].iat[0]), table["scenarios"].iat[0]
    kind = table.set_index("portfolio")
    cloud, curve = kind.loc[["random"]], kind.loc[["frontier"]]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(cloud["es"], cloud["upside"], s=8, color="#cfcec9", label="Random weightings")
        ax.plot(curve["es"], curve["upside"], color=VAR_99, linewidth=2,
                label="Frontier: least ES for each expected return")
        for name, color, side in (("current", BREACH, "left"), ("equal weight", INK_2, "right"),
                                  ("min ES", VAR_99, "left")):
            point = kind.loc[name]
            ax.scatter(point["es"], point["upside"], s=60, color=color, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(name, (point["es"], point["upside"]), ha=side, fontsize=9, color=color,
                        xytext=(8, -4) if side == "left" else (-6, 8), textcoords="offset points")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(f"Bad case: mean {days}-day loss in the worst 5% of scenarios (ES)")
        ax.set_ylabel("Good case: gain beaten only by the best 5%")
        ax.set_title("Same stocks, different weights: bad case vs. good case", pad=24)
        ax.text(0, 1.02, f"Every portfolio measured on the same scenarios; {scenarios}",
                transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--backtest", type=Path, default=REPORTS / "backtest.csv",
                        help="Per-date results written by backtest.py")
    parser.add_argument("--frontier", type=Path, default=REPORTS / "frontier.csv",
                        help="Portfolios written by frontier.py")
    parser.add_argument("--out", type=Path, default=REPORTS, help="Folder for the PNG files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        returns, start, shares, _ = load_inputs(args)
        values = simulate(returns, start, shares, args.paths, args.days, args.seed, args.drift,
                          **model_options(args))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    written = [paths_chart(values, args.out / "paths.png", args.days, seed=args.seed),
               distribution_chart(values, args.out / "distribution.png", args.days),
               correlation_chart(returns, args.out / "correlation.png")]
    if args.backtest.exists():
        results = pd.read_csv(args.backtest, index_col="date", parse_dates=["date", "end"])
        try:
            written.append(backtest_chart(results, args.out / "backtest.png"))
        except ValueError as error:
            print(f"Skipped backtest chart: {error}")
    else:
        print(f"Skipped backtest chart: {args.backtest} not found; run backtest.py first")
    if args.frontier.exists():
        written.append(frontier_chart(pd.read_csv(args.frontier), args.out / "frontier.png"))
    else:
        print(f"Skipped frontier chart: {args.frontier} not found; run frontier.py first")
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
