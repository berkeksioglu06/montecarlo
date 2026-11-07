import unittest

import numpy as np
import pandas as pd

from simulation import correlated_shocks, parameters


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


if __name__ == "__main__":
    unittest.main()
