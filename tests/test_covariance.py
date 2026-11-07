import unittest

import numpy as np
import pandas as pd

from backtest import backtest
from simulation import covariance, parameters, simulate
from test_backtest import SHARES, gbm_prices


def returns_frame(rows):
    return pd.DataFrame(rows, columns=["A", "B"], dtype=float)


class CovarianceTests(unittest.TestCase):
    def test_equal_is_the_sample_covariance(self):
        data = returns_frame(np.random.default_rng(1).normal(0, 0.02, (100, 2)))
        np.testing.assert_allclose(covariance(data, "equal"), data.cov())
        np.testing.assert_allclose(parameters(data)[1], data.cov())

    def test_ewma_matches_hand_computed_weights(self):
        data = returns_frame([[0.01, 0.02], [-0.03, 0.01], [0.02, -0.04]])  # oldest first
        lam = 0.5
        weights = np.array([0.25, 0.5, 1.0]) / 1.75  # newest row weighs most
        expected = sum(w * np.outer(row, row) for w, row in zip(weights, data.to_numpy()))
        np.testing.assert_allclose(covariance(data, "ewma", lam), expected)

    def test_ewma_reacts_to_a_recent_shock_faster(self):
        rng = np.random.default_rng(2)
        calm = rng.normal(0, 0.01, (240, 2))
        # The last 12 days move exactly 5% a day, alternating sign: five times more volatile.
        shock = np.tile([[0.05, -0.05], [-0.05, 0.05]], (6, 1))
        data = returns_frame(np.vstack([calm, shock]))
        equal_vol = np.sqrt(np.diag(covariance(data, "equal")))
        ewma_vol = np.sqrt(np.diag(covariance(data, "ewma")))
        # The last 12 days carry 1 - 0.94^12 = 52% of the EWMA weight but 12/252 = 5% of the equal
        # weight: expected vols about 3.7% vs. 1.5%.
        self.assertTrue((ewma_vol > 0.03).all())
        self.assertTrue((equal_vol < 0.02).all())
        self.assertTrue((ewma_vol > 2 * equal_vol).all())
        # Without a regime change both estimate the same volatility.
        quiet = returns_frame(rng.normal(0, 0.02, (2000, 2)))
        np.testing.assert_allclose(np.sqrt(np.diag(covariance(quiet, "ewma"))), 0.02, rtol=0.25)

    def test_ewma_rejects_bad_settings(self):
        data = returns_frame(np.zeros((70, 2)) + 0.01)
        with self.assertRaisesRegex(ValueError, "Unknown covariance"):
            covariance(data, "garch")
        for lam in (0.0, 1.0, 1.5):
            with self.assertRaisesRegex(ValueError, "lambda"):
                covariance(data, "ewma", lam)

    def test_simulation_and_backtest_use_the_chosen_covariance(self):
        prices = gbm_prices(n=150)
        returns = np.log(prices).diff().iloc[1:]
        equal = simulate(returns, prices.iloc[-1], SHARES, 200, 5, cov="equal")
        ewma = simulate(returns, prices.iloc[-1], SHARES, 200, 5, cov="ewma")
        self.assertFalse(np.allclose(equal, ewma))  # same seed, different covariance
        results = backtest(prices, SHARES, days=5, window=60, n_paths=100, step=5, cov="ewma")
        self.assertEqual(set(results["cov"]), {"ewma"})
        self.assertEqual(set(results["lam"]), {0.94})


if __name__ == "__main__":
    unittest.main()
