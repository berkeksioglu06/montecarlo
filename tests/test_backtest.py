import unittest

import numpy as np
import pandas as pd

from backtest import (backtest, breach_episodes, exceedance_summary, kupiec, non_overlapping,
                      summarize)


def gbm_prices(n=600, vols=(0.02, 0.03), seed=5):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, vols, (n, len(vols)))
    prices = 100 * np.exp(np.vstack([np.zeros(len(vols)), np.cumsum(log_returns, axis=0)]))
    index = pd.bdate_range("2020-01-01", periods=n + 1)
    return pd.DataFrame(prices, index=index, columns=["A", "B"][:len(vols)])


SHARES = pd.Series({"A": 3.0, "B": 2.0})


class BacktestTests(unittest.TestCase):
    def test_forecast_ignores_prices_after_its_date(self):
        prices = gbm_prices(n=200)
        base = backtest(prices, SHARES, days=5, window=60, n_paths=500, step=5)
        changed = prices.copy()
        t = base.index[3]
        changed.loc[changed.index > t] *= 0.5  # crash right after t
        moved = backtest(changed, SHARES, days=5, window=60, n_paths=500, step=5)
        self.assertEqual(base.loc[t, "var_95%"], moved.loc[t, "var_95%"])  # forecast unchanged
        self.assertGreater(moved.loc[t, "loss"], base.loc[t, "loss"])      # outcome changed
        self.assertTrue(moved.loc[t, "breach_99%"])

    def test_realized_loss_uses_fixed_shares_and_horizon_end(self):
        prices = gbm_prices(n=120)
        results = backtest(prices, SHARES, days=5, window=60, n_paths=200, step=7)
        t = results.index[0]
        end = prices.index[prices.index.get_loc(t) + 5]
        self.assertEqual(results.loc[t, "end"], end)
        expected = prices.loc[t] @ SHARES - prices.loc[end] @ SHARES
        self.assertAlmostEqual(results.loc[t, "loss"], expected)
        self.assertEqual(results.index[0], prices.index[60])  # first date with a full window

    def test_breach_rate_is_close_to_nominal_on_true_model(self):
        prices = gbm_prices(n=2500)
        results = backtest(prices, SHARES, days=5, window=252, n_paths=2_000, step=5)
        table = summarize(results, (0.95,), days=5, step=5)
        self.assertGreater(table.loc[0.95, "indep_n"], 400)
        self.assertGreater(table.loc[0.95, "p_value"], 0.01)

    def test_kupiec(self):
        self.assertEqual(kupiec(100, 5, 0.95), (0.0, 1.0))
        lr, p_value = kupiec(250, 0, 0.99)  # zero breaches is valid input
        self.assertAlmostEqual(lr, -2 * 250 * np.log(0.99))
        self.assertGreater(p_value, 0.01)
        lr, p_value = kupiec(100, 20, 0.95)
        self.assertLess(p_value, 1e-5)
        with self.assertRaises(ValueError):
            kupiec(10, 11, 0.95)

    def test_non_overlapping_windows_do_not_share_days(self):
        prices = gbm_prices(n=150)
        results = backtest(prices, SHARES, days=6, window=60, n_paths=100, step=2)
        picked = non_overlapping(results, days=6, step=2)
        self.assertTrue((picked["end"].iloc[:-1].to_numpy() <= picked.index[1:].to_numpy()).all())
        with self.assertRaisesRegex(ValueError, "multiple"):
            non_overlapping(results, days=6, step=4)

    def test_rejects_short_history_and_empty_portfolio(self):
        prices = gbm_prices(n=70)
        with self.assertRaisesRegex(ValueError, "more than"):
            backtest(prices, SHARES, days=21, window=60)
        with self.assertRaisesRegex(ValueError, "no positions"):
            backtest(prices, pd.Series(dtype=float), days=5, window=60)


    def test_es_is_recorded_and_not_below_var(self):
        results = backtest(gbm_prices(n=120), SHARES, days=5, window=60, n_paths=500, step=5)
        for level in ("95%", "99%"):
            self.assertTrue((results[f"es_{level}"] >= results[f"var_{level}"]).all())

    def test_breach_episodes_merge_overlapping_windows_only(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        results = pd.DataFrame({"end": dates.shift(5), "value": 100.0, "loss": 1.0,
                                "var_95%": 10.0, "es_95%": 12.0, "breach_95%": False}, index=dates)
        # Breaches on days 0-2 overlap; day 5 starts exactly where day 0 ends but day 2's window
        # runs to day 7, so it still joins; day 20 is far away.
        for day, loss in ((0, 11.0), (1, 15.0), (2, 12.0), (5, 13.0), (20, 30.0)):
            results.iloc[day, results.columns.get_loc("loss")] = loss
            results.iloc[day, results.columns.get_loc("breach_95%")] = True
        episodes = breach_episodes(results)
        self.assertEqual(episodes["breaches"].tolist(), [4, 1])
        self.assertEqual(episodes.loc[0, "worst"], dates[1])
        self.assertAlmostEqual(episodes.loc[0, "excess"], 0.05)
        self.assertAlmostEqual(episodes.loc[1, "loss_to_var"], 3.0)
        # A breach starting on the day all earlier windows end shares no returns: new episode.
        results.iloc[7, results.columns.get_loc("breach_95%")] = True
        results.iloc[7, results.columns.get_loc("loss")] = 11.0
        results.iloc[5, results.columns.get_loc("breach_95%")] = False
        self.assertEqual(breach_episodes(results)["breaches"].tolist(), [3, 1, 1])
        summary = exceedance_summary(results)
        self.assertEqual((summary["breaches"], summary["episodes"]), (5, 3))
        self.assertAlmostEqual(summary["mean_es"], 0.12)

    def test_no_breaches_gives_empty_episodes(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        results = pd.DataFrame({"end": dates.shift(5), "value": 1.0, "loss": 0.0, "var_95%": 1.0,
                                "es_95%": 1.0, "breach_95%": False}, index=dates)
        self.assertTrue(breach_episodes(results).empty)
        self.assertEqual(exceedance_summary(results), {"breaches": 0})


if __name__ == "__main__":
    unittest.main()
