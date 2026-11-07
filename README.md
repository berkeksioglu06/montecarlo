# montecarlo

Portfolio risk simulation, built step by step: price collection, daily log returns, correlated GBM simulation of a fixed-share portfolio, Monte Carlo risk metrics and a rolling VaR backtest, with charts.

## Setup and download

Python 3.12+:

```sh
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python market_data.py
```

Default symbols: ZETA, RKLB, PLTR, VLN, HIMS, MRVL, ACHR, EOSE. Requested history starts on 2021-01-01; shorter histories are reported. Override with `--start YYYY-MM-DD` or `--symbols TICKER ...`.

Use `--as-of YYYY-MM-DD` to request a specific completed trading session. Without it, the latest completed session is required; incomplete provider bars fail validation rather than causing a silent fallback.

Each run creates `data/yahoo_<UTC timestamp>/` with:

- `raw/`: unchanged provider responses, including corporate actions.
- `clean/`: dates normalized, exact duplicate rows and incomplete sessions removed.
- `manifest.json`: settings, versions, currency, SHA-256 checksums and run status.
- `issues.csv` and `coverage.csv`: quality findings and per-symbol date coverage.
- `prices.csv`: aligned adjusted closes on the common trading calendar, only when all symbols pass validation.
- `close_prices.csv`: the corresponding split-adjusted Close series, without dividend adjustment.

No missing prices are filled. Conflicting duplicates, missing common sessions, stale data, non-USD currency and invalid prices block aligned output. Moves above 25% and split dates are flagged for review; passing checks does not independently certify corporate-action adjustments. Pre-merger/SPAC history needs review before choosing the estimation window.

Adjusted Close includes dividend adjustments and is a total-return proxy. Fixed-share price simulation and dividend handling must be chosen explicitly at the modeling step. User-supplied portfolio snapshot prices are separate from historical calibration data.

## Tests

```sh
.venv\Scripts\python -m unittest discover -s tests -v
```

## Source

Adapted from [quant-ml/market_data.py](https://github.com/berkeksioglu06/quant-ml/blob/dd220ad388ca6c3a3e04ce5b671d5522d3fb693f/market_data.py): explicit Yahoo settings, completed-session cutoff, snapshot storage and quality checks. This repository runs independently; ML features, labels and training folds are excluded. Local datasets and personal holdings are ignored by Git.

The GitHub Actions workflow runs tests and downloads all eight symbols when this data code changes, or when manually triggered. It commits a snapshot only after all symbols pass validation with at least 252 common sessions. Failed downloads leave diagnostics as a workflow artifact, not a usable dataset. There is no scheduled refresh.

The initial workflow snapshot is explicitly pinned to 2026-11-05: Yahoo returned missing Close and Adj Close fields for 2026-11-06 during the first download. Manual workflow runs can specify another as-of date.

## Simulation and risk

Copy `portfolio.example.csv` to `portfolio.csv` (ignored by Git) and fill in share counts. The optional `price` column sets start prices; blank uses the snapshot's last close.

```sh
.venv\Scripts\python returns.py      # volatility, tails, correlation
.venv\Scripts\python simulation.py   # distribution of portfolio value after 21 days
.venv\Scripts\python risk.py         # VaR, ES, drawdown, variance shares
.venv\Scripts\python backtest.py     # rolling 21-day VaR vs realized loss (~1 min)
.venv\Scripts\python charts.py       # PNG charts in reports/
```

- `returns.py`: daily log returns from the latest READY snapshot's adjusted closes.
- `simulation.py`: Cholesky-correlated normal log returns; prices compound as GBM with fixed shares. `--drift zero` (default) sets the mean log return to zero, which is not a zero expected return; `--drift historical` uses the sample mean.
- `risk.py`: losses are `L = V0 - VT` (positive is a loss). VaR is the mildest loss in the worst `ceil(N(1 - level))` outcomes and ES is their mean. Parametric VaR is a delta-normal approximation, not an exact check; the exact check is the single-stock GBM test. Variance shares split daily portfolio variance, not the 21-day VaR.
- `backtest.py`: at each date t the model is calibrated on the last 252 returns ending at t and starts from the close at t; nothing after t is used. A breach is a realized loss `V_t - V_{t+21}` above that VaR, with the same fixed shares. Daily forecasts overlap and their breaches cluster, so the Kupiec test uses only non-overlapping 21-day windows (about 48 here, which gives the test little power, especially at 99%). Per-date results go to `reports/backtest.csv`. Historical adjusted closes are the start prices; the portfolio's price column is ignored. The eight symbols were chosen today, so the history is not free of selection bias.
- `charts.py`: `reports/paths.png` (a sample of simulated value paths with percentile bands from all paths), `reports/backtest.png` (VaR lines against realized loss, breaches marked; needs `backtest.py` first), `reports/distribution.png` (simulated outcomes with VaR and ES) and `reports/correlation.png`. Losses are shown as a share of the portfolio value at each date.

Known limitation: normal shocks understate these stocks' fat tails (historical excess kurtosis 3 to 15).
