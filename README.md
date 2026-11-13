# montecarlo

Portfolio risk simulation, built step by step: price collection, daily log returns, correlated GBM simulation of a fixed-share portfolio, Monte Carlo risk metrics and a rolling VaR backtest, with charts. On top of that measurement layer sits the research question: keeping the same stocks and changing only their weights, how much can the 21-day bad-case loss be cut, and how much upside survives?

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
- `simulation.py`: Cholesky-correlated normal log returns; prices compound as GBM with fixed shares. `--drift zero` (default) sets the mean log return to zero, which is not a zero expected return; `--drift historical` uses the sample mean. `--cov ewma` (all scripts, including the backtest) replaces the equal-weight sample covariance with a RiskMetrics exponentially weighted one (`--lam`, default 0.94: a day's weight halves in about 11 trading days). It reacts to new shocks within days, but it holds today's covariance fixed over the whole horizon, so after a calm spell it also lowers the 21-day VaR quickly. `--shocks t` (all scripts) draws multivariate Student-t shocks with the same covariance (`--nu`, default 5: excess kurtosis 6 per stock). Each path-day shares one chi-squared draw across all stocks, so large moves arrive together even between uncorrelated stocks. With the variance held fixed, t moves probability into the center and the far tail: it lowers the 95% VaR and raises the 99% VaR.
- `risk.py`: losses are `L = V0 - VT` (positive is a loss). VaR is the mildest loss in the worst `ceil(N(1 - level))` outcomes and ES is their mean. Parametric VaR is a delta-normal approximation, not an exact check; the exact check is the single-stock GBM test. Variance shares split daily portfolio variance, not the 21-day VaR.
- `backtest.py`: at each date t the model is calibrated on the last 252 returns ending at t and starts from the close at t; nothing after t is used. A breach is a realized loss `V_t - V_{t+21}` above that VaR, with the same fixed shares. Daily forecasts overlap and their breaches cluster, so the Kupiec test uses only non-overlapping 21-day windows (about 48 here, which gives the test little power; at 99% fewer than half a breach is expected, so the 95% level is the one to read). For 95% it also groups breaches into episodes of overlapping windows and reports how far losses went past VaR, and the mean breach loss against the model's own ES on those dates. The Christoffersen test checks whether one breach makes the next more likely; it only looks at consecutive forecasts, so slower clustering over weeks can pass it. `--days 1 --out reports/backtest_1d.csv` runs one-day forecasts, which share no return days, so every forecast counts for the tests (about 1,000 instead of 48). Per-date results go to `reports/backtest.csv`. Historical adjusted closes are the start prices; the portfolio's price column is ignored. The eight symbols were chosen today, so the history is not free of selection bias.
- `charts.py`: `reports/paths.png` (a sample of simulated value paths with percentile bands from all paths), `reports/backtest.png` (VaR lines against realized loss, breaches marked; needs `backtest.py` first), `reports/distribution.png` (simulated outcomes with VaR and ES) and `reports/correlation.png`. Losses are shown as a share of the portfolio value at each date.

Fat tails: these stocks' daily returns have excess kurtosis 3 to 15, which normal shocks miss. Backtest share of forecasts breaching VaR (1-day: 1,028 forecasts, all independent; 21-day: 1,008 overlapping, Kupiec on 48), with nu = 5 set once beforehand and not tuned on the backtest:

| horizon | covariance | shocks | 95% (expect 5%) | 99% (expect 1%) |
|---|---|---|---|---|
| 1 day | equal | normal | 4.8% (p 0.73) | 1.8% (p 0.03) |
| 1 day | equal | t, nu 5 | 5.9% (p 0.18) | 1.1% (p 0.82) |
| 1 day | EWMA | normal | 5.8% (p 0.23) | 2.0% (p 0.003) |
| 1 day | EWMA | t, nu 5 | 6.7% (p 0.02) | 1.4% (p 0.27) |
| 21 days | equal | normal | 3.0% | 1.0% |
| 21 days | equal | t, nu 5 | 3.1% | 1.0% |
| 21 days | EWMA | normal | 4.2% | 1.2% |
| 21 days | EWMA | t, nu 5 | 4.3% | 1.0% |

At one day, normal shocks are rejected at 99% and t fixes it, at the cost of a lower 95% VaR. Over 21 days t changes almost nothing: a sum of 21 independent daily shocks has 1/21 of the daily excess kurtosis, so it is close to normal whatever the daily shape. Monthly fat tails in real prices come mostly from volatility that changes over time, which neither model simulates within the horizon.

## Reweighting: less bad case, how much upside?

Same stocks, long only, weights sum to 100%, no trading inside the 21 days. Only the weights change.

```sh
.venv\Scripts\python frontier.py      # frontier and the current portfolio's distance from it (~20 s)
.venv\Scripts\python walkforward.py   # out-of-sample test (~30 s)
.venv\Scripts\python charts.py        # adds reports/frontier.png
```

- Bad case: 95% expected shortfall (ES), the mean 21-day loss in the worst 5% of scenarios. Good case ("upside"): the 21-day gain that only the best 5% of scenarios beat. Both are shares of today's value.
- `frontier.py` simulates the 21-day return of every stock once and judges every set of weights on those same scenarios, so portfolios differ by their weights and not by simulation luck. For a buy-and-hold portfolio the return is exactly the weighted sum of the stock returns. Minimum-ES weights come from the Rockafellar-Uryasev linear program, so the optimum is global. The frontier is the least ES for each expected-return target, from the safest mix up to the single highest-mean stock. The mean is the target because it is the only reward that keeps the problem linear; the upside is then measured on each frontier portfolio. 1,000 random weightings are drawn as a check: none reaches more upside than the frontier at the same ES by more than about half a point.
- Scenarios come from the GBM model (`--source gbm`, default) or from a block bootstrap of real days (`--source bootstrap`, `--block 5`), which keeps joint crashes, skew and week-long volatility clustering. `--drift zero` centers every stock's mean log return at zero in both sources; `--drift historical` keeps the sample mean. That gives a 2 x 2 comparison of drift assumption and return shape.
- `walkforward.py`: every 21 trading days the weights are re-optimized on the last 252 returns only, held for 21 days and scored on the actual closes. Four strategies: today's weights (applied at every date), equal weight, minimum ES, and minimum ES with at least the model mean of today's weights.

What the model says today (10,000 scenarios, full history; ES cut and upside kept relative to the current weights):

| scenarios | drift | min ES: ES cut | min ES: upside kept | equal weight: ES cut | equal weight: upside kept |
|---|---|---|---|---|---|
| GBM | zero | 13% | 88% | 8% | 99% |
| GBM | historical | 12% | 88% | 5% | 92% |
| bootstrap | zero | 11% | 89% | 7% | 99% |
| bootstrap | historical | 9% | 85% | 3% | 93% |

What happened out of sample (48 non-overlapping 21-day periods, 2022-09 to 2026-10):

| strategy | mean | realized ES | realized upside | ES cut | upside kept |
|---|---|---|---|---|---|
| current weights | 6.1% | 19.9% | 27.1% | | |
| equal weight | 5.9% | 18.4% | 40.5% | 8% | 149% |
| min ES at same mean | 5.5% | 17.7% | 27.0% | 11% | 100% |
| min ES | 5.5% | 17.6% | 26.3% | 11% | 97% |

Reading it:

- The cut is real but modest: about a tenth of the bad case, in the model and out of sample. The stocks are all volatile and move together, so reweighting cannot diversify much away; the concentration costs less than the weights suggest.
- In the model the upside falls almost as fast as the bad case (ES down 13%, upside down 12% under zero drift). Adding historical drift or real return shapes does not change that, so the answer is not an artifact of the symmetric model.
- Run with a rebalance every 3 days instead (`--step 3`: 336 overlapping periods, so every start date counts), min ES cut the realized ES by 12% and kept 89% of the upside, almost exactly what the model predicts. Equal weight cut nothing there (-2%) and kept 110%: its good showing on the 48 periods depends on where the windows start.
- With other scenario settings the 48-period ES cut of min ES ranges from 3% to 15%. Realized ES and upside rest on three periods each out of 48, so differences of a point or two are noise.

Limits: the eight stocks were chosen today and today's weights reflect past winners; no trading costs or taxes, and the min-ES weights move a lot between rebalances; volatility is constant inside the horizon; different seeds move individual min-ES weights by a few points because ES is flat near the optimum.
