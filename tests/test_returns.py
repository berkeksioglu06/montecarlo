import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from returns import latest_snapshot, load_prices, log_returns, summary


def prices(values):
    dates = pd.bdate_range("2026-11-02", periods=len(values))
    return pd.DataFrame({"AAA": values}, index=dates.rename("date"), dtype=float)


class ReturnsTests(unittest.TestCase):
    def test_known_log_returns(self):
        result = log_returns(prices([100, 110, 99]))["AAA"]
        np.testing.assert_allclose(result, [np.log(1.1), np.log(0.9)])

    def test_daily_log_returns_add_to_total(self):
        series = prices([10, 12, 9, 15, 14])
        self.assertAlmostEqual(log_returns(series)["AAA"].sum(), np.log(14 / 10))

    def test_summary_annualizes_mean_and_vol(self):
        returns = pd.DataFrame({"AAA": [0.01, -0.01] * 50})
        row = summary(returns).loc["AAA"]
        self.assertAlmostEqual(row["ann_mean"], 0.0)
        self.assertAlmostEqual(row["ann_vol"], returns["AAA"].std() * np.sqrt(252))

    def test_load_rejects_non_positive_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices([10, 0, 11]).to_csv(Path(tmp) / "prices.csv")
            with self.assertRaisesRegex(ValueError, "positive"):
                load_prices(Path(tmp))

    def test_latest_snapshot_skips_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, status in [("yahoo_20260101T000000Z", "READY"),
                                 ("yahoo_20260201T000000Z", "FAILED")]:
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / "manifest.json").write_text(json.dumps({"status": status}))
            self.assertEqual(latest_snapshot(Path(tmp)).name, "yahoo_20260101T000000Z")


if __name__ == "__main__":
    unittest.main()
