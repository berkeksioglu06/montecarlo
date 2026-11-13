import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import backtest
from charts import (backtest_chart, backtest_labels, correlation_chart, distribution_chart,
                    frontier_chart, paths_chart)
from frontier import compare, frontier, random_mixes, scenarios
from simulation import simulate
from test_backtest import SHARES, gbm_prices

PNG = b"\x89PNG"


class ChartTests(unittest.TestCase):
    def test_each_chart_writes_a_png(self):
        prices = gbm_prices(n=150)
        returns = np.log(prices).diff().iloc[1:]
        values = simulate(returns, prices.iloc[-1], SHARES, n_paths=500, n_days=5)
        results = backtest(prices, SHARES, days=5, window=60, n_paths=200, step=5)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [backtest_chart(results, Path(tmp) / "b.png"),
                     distribution_chart(values, Path(tmp) / "d.png", days=5),
                     paths_chart(values, Path(tmp) / "p.png", days=5, n_show=50),
                     correlation_chart(returns, Path(tmp) / "c.png")]
            for path in paths:
                self.assertEqual(path.read_bytes()[:4], PNG)

    def test_frontier_chart_reads_saved_csv(self):
        scen = scenarios(np.log(gbm_prices(n=150)).diff().iloc[1:], 500, 5)
        table = pd.concat([compare(scen, pd.Series({"A": 0.1, "B": 0.9})).rename_axis("portfolio").reset_index(),
                           frontier(scen, 4).assign(portfolio="frontier"),
                           random_mixes(scen, 20).assign(portfolio="random")])
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "frontier.csv"
            table.assign(days=5, scenarios="test").to_csv(csv, index=False)
            path = frontier_chart(pd.read_csv(csv), Path(tmp) / "f.png")
            self.assertEqual(path.read_bytes()[:4], PNG)

    def test_backtest_chart_reads_saved_csv(self):
        # The chart must work on what backtest.py writes, not only on the in-memory frame.
        results = backtest(gbm_prices(n=120), SHARES, days=5, window=60, n_paths=200, step=5)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "backtest.csv"
            results.to_csv(csv)
            loaded = pd.read_csv(csv, index_col="date", parse_dates=["date", "end"])
            self.assertEqual(backtest_chart(loaded, Path(tmp) / "b.png").read_bytes()[:4], PNG)

    def test_backtest_labels_come_from_the_csv_settings(self):
        results = backtest(gbm_prices(n=200), SHARES, days=7, window=90, n_paths=100, step=7)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "backtest.csv"
            results.to_csv(csv)
            loaded = pd.read_csv(csv, index_col="date", parse_dates=["date", "end"])
        labels = backtest_labels(loaded)
        self.assertEqual(labels["title"], "Rolling 7-day VaR vs. realized loss")
        self.assertEqual(labels["loss"], "Realized 7-day loss")
        self.assertIn("last 90 daily returns", labels["subtitle"])
        self.assertIn("next 7 trading days", labels["subtitle"])
        self.assertNotIn("252", labels["subtitle"])
        self.assertNotIn("21", " ".join(labels.values()))

    def test_backtest_labels_reject_unknown_or_mixed_settings(self):
        results = backtest(gbm_prices(n=120), SHARES, days=5, window=60, n_paths=100, step=5)
        with self.assertRaisesRegex(ValueError, "rerun backtest.py"):
            backtest_labels(results.drop(columns=["days", "window"]))  # CSV from before this fix
        mixed = results.copy()
        mixed.iloc[0, mixed.columns.get_loc("window")] = 252
        with self.assertRaisesRegex(ValueError, "mix several window"):
            backtest_labels(mixed)


if __name__ == "__main__":
    unittest.main()
