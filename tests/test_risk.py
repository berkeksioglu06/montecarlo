import unittest

import numpy as np
import pandas as pd

from risk import gbm_var_es, losses, max_drawdown, parametric_var, var_es, variance_contribution
from simulation import simulate


def random_returns(n=500, symbols=("A", "B")):
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 0.02, (n, len(symbols)))
    data[:, -1] += 0.5 * data[:, 0]  # some correlation
    return pd.DataFrame(data, columns=list(symbols))


class RiskTests(unittest.TestCase):
    def test_losses_are_positive_when_value_falls(self):
        values = np.array([[100.0, 90.0, 80.0], [100.0, 110.0, 120.0]])
        np.testing.assert_allclose(losses(values), [20, -20])
        np.testing.assert_allclose(losses(values, day=1), [10, -10])

    def test_var_es_use_the_same_worst_tail(self):
        loss = np.arange(1.0, 101.0)  # 1..100
        var, es = var_es(loss, 0.95)
        self.assertEqual(var, 96.0)  # worst five: 96..100
        self.assertEqual(es, 98.0)
        var, es = var_es(np.full(100, 7.0), 0.99)
        self.assertEqual((var, es), (7.0, 7.0))  # ES equals VaR when the tail is flat
        for level in (0.0, 1.0, 95):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                var_es(loss, level)

    def test_simulation_rejects_empty_horizon(self):
        returns = random_returns(symbols=("A",))
        for paths, days in ((10, 0), (0, 21)):
            with self.assertRaisesRegex(ValueError, "positive"):
                simulate(returns, pd.Series({"A": 1.0}), pd.Series({"A": 1.0}), paths, days)

    def test_max_drawdown_counts_from_running_peak(self):
        values = np.array([[100.0, 120.0, 90.0, 130.0], [100.0, 90.0, 95.0, 80.0]])
        np.testing.assert_allclose(max_drawdown(values), [0.25, 0.20])

    def test_simulation_matches_exact_single_stock_gbm(self):
        returns = random_returns(symbols=("A",))
        vol = returns["A"].std()
        for drift, mean in (("zero", 0.0), ("historical", returns["A"].mean())):
            values = simulate(returns, pd.Series({"A": 50.0}), pd.Series({"A": 2.0}),
                              n_paths=200_000, n_days=21, seed=11, drift=drift)
            for level in (0.95, 0.99):
                exact_var, exact_es = gbm_var_es(100.0, mean, vol, 21, level)
                var, es = var_es(losses(values), level)
                self.assertAlmostEqual(var, exact_var, delta=0.02 * exact_var)
                self.assertAlmostEqual(es, exact_es, delta=0.02 * exact_es)

    def test_variance_shares_sum_to_one_and_follow_symbols(self):
        returns = random_returns()
        start, shares = pd.Series({"B": 10.0, "A": 20.0}), pd.Series({"A": 1.0, "B": 2.0})
        table = variance_contribution(returns, start, shares)
        self.assertAlmostEqual(table["variance_share"].sum(), 1.0)
        np.testing.assert_allclose(table["weight"], [0.5, 0.5])

    def test_parametric_var_scales_with_square_root_of_time(self):
        returns = random_returns()
        start, shares = pd.Series({"A": 10.0, "B": 10.0}), pd.Series({"A": 1.0, "B": 1.0})
        one = parametric_var(returns, start, shares, 0.95, 1)
        self.assertAlmostEqual(parametric_var(returns, start, shares, 0.95, 4), 2 * one)


if __name__ == "__main__":
    unittest.main()
