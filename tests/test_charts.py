import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import backtest
from charts import backtest_chart, correlation_chart, distribution_chart, paths_chart
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
            paths = [backtest_chart(results, Path(tmp) / "b.png", days=5),
                     distribution_chart(values, Path(tmp) / "d.png", days=5),
                     paths_chart(values, Path(tmp) / "p.png", days=5, n_show=50),
                     correlation_chart(returns, Path(tmp) / "c.png")]
            for path in paths:
                self.assertEqual(path.read_bytes()[:4], PNG)

    def test_backtest_chart_reads_saved_csv(self):
        # The chart must work on what backtest.py writes, not only on the in-memory frame.
        results = backtest(gbm_prices(n=120), SHARES, days=5, window=60, n_paths=200, step=5)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "backtest.csv"
            results.to_csv(csv)
            loaded = pd.read_csv(csv, index_col="date", parse_dates=["date", "end"])
            self.assertEqual(backtest_chart(loaded, Path(tmp) / "b.png").read_bytes()[:4], PNG)


if __name__ == "__main__":
    unittest.main()
