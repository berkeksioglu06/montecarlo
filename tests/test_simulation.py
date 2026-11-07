import unittest

import numpy as np
import pandas as pd

import tempfile
from pathlib import Path

from simulation import (correlated_shocks, load_portfolio, parameters, portfolio_values,
                        price_paths)


class SimulationTests(unittest.TestCase):
    def test_shocks_reproduce_target_covariance(self):
        cov = pd.DataFrame([[0.04, 0.012], [0.012, 0.01]], index=["A", "B"], columns=["A", "B"])
        shocks = correlated_shocks(cov, 200_000, 1, np.random.default_rng(0)).reshape(-1, 2)
        np.testing.assert_allclose(np.cov(shocks.T), cov, rtol=0.02)

    def test_zero_mean_is_default(self):
        returns = pd.DataFrame({"A": [0.01, 0.03], "B": [-0.02, 0.0]})
        mean, cov = parameters(returns)
        self.assertTrue((mean == 0).all())
        np.testing.assert_allclose(parameters(returns, zero_mean=False)[0], [0.02, -0.01])
        np.testing.assert_allclose(cov, returns.cov())

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

    def test_portfolio_aligns_to_price_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portfolio.csv"
            path.write_text("symbol,shares\nB,5\nA,2\n")
            shares = load_portfolio(path, pd.Index(["A", "B", "C"]))
            self.assertEqual(shares.tolist(), [2.0, 5.0, 0.0])
            path.write_text("symbol,shares\nA,2\nZZZ,1\n")
            with self.assertRaisesRegex(ValueError, "ZZZ"):
                load_portfolio(path, pd.Index(["A"]))
            path.write_text("symbol,shares\nA,-1\n")
            with self.assertRaisesRegex(ValueError, "positive"):
                load_portfolio(path, pd.Index(["A"]))


if __name__ == "__main__":
    unittest.main()
