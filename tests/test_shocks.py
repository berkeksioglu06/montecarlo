import unittest

import numpy as np
import pandas as pd

from backtest import backtest
from simulation import correlated_shocks, simulate
from test_backtest import SHARES, gbm_prices

COV = pd.DataFrame([[4e-4, 1.2e-4], [1.2e-4, 1e-4]], index=["A", "B"], columns=["A", "B"])


def draws(shocks, nu=5.0, n=400_000, cov=COV, seed=0):
    return correlated_shocks(cov, n, 1, np.random.default_rng(seed), shocks, nu).reshape(-1, len(cov))


def excess_kurtosis(x):
    x = x - x.mean()
    return (x ** 4).mean() / (x ** 2).mean() ** 2 - 3


class StudentShockTests(unittest.TestCase):
    def test_t_shocks_keep_the_target_covariance(self):
        np.testing.assert_allclose(np.cov(draws("t").T), COV, rtol=0.03)

    def test_t_shocks_have_fat_tails(self):
        # Excess kurtosis of a Student-t is 6 / (nu - 4): 1.0 at nu = 10, 0 for the normal.
        self.assertAlmostEqual(excess_kurtosis(draws("t", nu=10)[:, 0]), 1.0, delta=0.25)
        self.assertAlmostEqual(excess_kurtosis(draws("normal")[:, 0]), 0.0, delta=0.05)

    def test_large_moves_arrive_together_even_without_correlation(self):
        # Uncorrelated is not independent: the shared chi-squared draw makes calm and wild days
        # common to all stocks, so squared shocks correlate under t but not under the normal.
        identity = pd.DataFrame(np.eye(2) * 1e-4)
        normal, student = draws("normal", cov=identity), draws("t", cov=identity)
        self.assertLess(abs(np.corrcoef(normal.T)[0, 1]), 0.01)
        self.assertLess(abs(np.corrcoef(student.T)[0, 1]), 0.01)
        self.assertLess(abs(np.corrcoef(normal.T ** 2)[0, 1]), 0.01)
        self.assertGreater(np.corrcoef(student.T ** 2)[0, 1], 0.1)

    def test_same_variance_moves_risk_into_the_far_tail(self):
        # With the variance held fixed, t puts more mass near zero and in the far tail, so its 95%
        # quantile is below the normal one and its 99.9% quantile above it.
        normal, student = draws("normal")[:, 0], draws("t", nu=4)[:, 0]
        self.assertLess(np.quantile(student, 0.95), np.quantile(normal, 0.95))
        self.assertGreater(np.quantile(student, 0.999), np.quantile(normal, 0.999))

    def test_t_rescales_the_same_normal_draws(self):
        identity = pd.DataFrame(np.eye(3) * 1e-4)
        normal = correlated_shocks(identity, 50, 4, np.random.default_rng(3))
        student = correlated_shocks(identity, 50, 4, np.random.default_rng(3), "t", 5)
        ratio = student / normal  # one positive scale per path-day, shared by every stock
        np.testing.assert_allclose(ratio, np.repeat(ratio[..., :1], 3, axis=2))
        self.assertTrue((ratio > 0).all())
        np.testing.assert_array_equal(normal, correlated_shocks(identity, 50, 4,
                                                                np.random.default_rng(3), "normal"))

    def test_bad_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nu > 2"):
            draws("t", nu=2, n=10)
        with self.assertRaisesRegex(ValueError, "Unknown shocks"):
            draws("laplace", n=10)

    def test_simulation_and_backtest_use_the_chosen_shocks(self):
        prices = gbm_prices(n=150)
        returns = np.log(prices).diff().iloc[1:]
        normal = simulate(returns, prices.iloc[-1], SHARES, 200, 5)
        student = simulate(returns, prices.iloc[-1], SHARES, 200, 5, shocks="t", nu=6)
        self.assertFalse(np.allclose(normal, student))
        results = backtest(prices, SHARES, days=5, window=60, n_paths=100, step=5, shocks="t", nu=6)
        self.assertEqual(set(results["shocks"]), {"t"})
        self.assertEqual(set(results["nu"]), {6})
        plain = backtest(prices, SHARES, days=5, window=60, n_paths=100, step=5)
        self.assertEqual(set(plain["shocks"]), {"normal"})
        self.assertTrue(plain["nu"].isna().all())


if __name__ == "__main__":
    unittest.main()
