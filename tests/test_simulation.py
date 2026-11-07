import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import (correlated_shocks, load_portfolio, parameters, portfolio_values,
                        price_paths, simulate, start_prices)


def random_returns(n=100, symbols=("A", "B")):
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(0.001, 0.02, (n, len(symbols))), columns=list(symbols))


class SimulationTests(unittest.TestCase):
    def test_shocks_reproduce_target_covariance(self):
        cov = pd.DataFrame([[0.04, 0.012], [0.012, 0.01]], index=["A", "B"], columns=["A", "B"])
        shocks = correlated_shocks(cov, 200_000, 1, np.random.default_rng(0)).reshape(-1, 2)
        np.testing.assert_allclose(np.cov(shocks.T), cov, rtol=0.02)

    def test_zero_log_drift_is_default(self):
        returns = random_returns()
        mean, cov = parameters(returns)
        self.assertTrue((mean == 0).all())
        np.testing.assert_allclose(parameters(returns, "historical")[0], returns.mean())
        np.testing.assert_allclose(cov, returns.cov())
        with self.assertRaisesRegex(ValueError, "Unknown drift"):
            parameters(returns, "mean")

    def test_insufficient_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least"):
            parameters(random_returns(n=1))

    def test_singular_covariance_is_rejected(self):
        cov = pd.DataFrame([[1e-4, 1e-4], [1e-4, 1e-4]])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            correlated_shocks(cov, 1, 1, np.random.default_rng(0))

    def test_same_seed_same_paths(self):
        cov = pd.DataFrame(np.eye(3) * 1e-4)
        a = correlated_shocks(cov, 10, 5, np.random.default_rng(42))
        b = correlated_shocks(cov, 10, 5, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (10, 5, 3))

    def test_price_paths_compound_log_returns(self):
        start = pd.Series([100.0, 10.0], index=["A", "B"])
        shocks = np.log([[[1.1, 1.0], [0.9, 2.0]]])  # one path, two days
        paths = price_paths(start, pd.Series(0.0, index=start.index), shocks)
        np.testing.assert_allclose(paths[0], [[100, 10], [110, 10], [99, 20]])

    def test_portfolio_values_use_fixed_shares(self):
        paths = np.array([[[100.0, 10.0], [110.0, 20.0]]])
        values = portfolio_values(paths, pd.Series([2.0, 5.0]))
        np.testing.assert_allclose(values, [[250, 320]])

    def test_simulate_aligns_by_symbol_not_position(self):
        returns = random_returns()
        start = pd.Series({"B": 10.0, "A": 100.0})
        shares = pd.Series({"B": 5.0, "A": 2.0})
        values = simulate(returns, start, shares, n_paths=5, n_days=3)
        self.assertAlmostEqual(values[0, 0], 250.0)
        np.testing.assert_allclose(values, simulate(returns, start[["A", "B"]],
                                                    shares[["A", "B"]], n_paths=5, n_days=3))
        with self.assertRaisesRegex(ValueError, "ZZZ"):
            simulate(returns, start, pd.Series({"ZZZ": 1.0}))

    def test_portfolio_prices_override_last_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portfolio.csv"
            path.write_text("symbol,shares,price\nA,2,120\nB,5,\n")
            portfolio = load_portfolio(path)
            start = start_prices(pd.Series({"A": 100.0, "B": 10.0}), portfolio)
            self.assertEqual(start.to_dict(), {"A": 120.0, "B": 10.0})
            path.write_text("symbol,shares\nA,-1\n")
            with self.assertRaisesRegex(ValueError, "positive"):
                load_portfolio(path)


if __name__ == "__main__":
    unittest.main()
