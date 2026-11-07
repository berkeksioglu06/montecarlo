# montecarlo

Portfolio risk simulation, built step by step. Current step: historical price collection and validation. The GBM simulator is not implemented yet.

## Setup and download

Python 3.12+:

```sh
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python market_data.py
```

Default symbols: ZETA, RKLB, PLTR, VLN, HIMS, MRVL, ACHR, EOSE. Requested history starts on 2021-01-01; shorter histories are reported. Override with `--start YYYY-MM-DD` or `--symbols TICKER ...`.

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

Next: choose the estimation window, calculate daily log returns and covariance, then simulate 10,000 correlated GBM paths over 21 trading days.
