import unittest

import numpy as np
import pandas as pd

from frontier import measure, min_es_weights, scenarios
from returns import log_returns
from test_backtest import gbm_prices
from walkforward import walk_forward
from weights import model_weights


class ModelTests(unittest.TestCase):
    def test_weights_are_a_portfolio_and_the_calm_stock_gets_most(self):
        returns = log_returns(gbm_prices(n=400, vols=(0.01, 0.03)))
        weights, scen = model_weights(returns, 2_000, 10)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertTrue((weights >= 0).all())
        self.assertGreater(weights["A"], 0.8)
        equal = pd.Series(0.5, index=weights.index)
        self.assertLess(measure(scen, weights)["es"], measure(scen, equal)["es"])

    def test_model_is_minimum_es_on_its_own_scenarios(self):
        returns = log_returns(gbm_prices(n=300))
        weights, scen = model_weights(returns, 1_000, 5, seed=4, source="bootstrap", block=3)
        pd.testing.assert_frame_equal(scen, scenarios(returns, 1_000, 5, 4, source="bootstrap", block=3))
        pd.testing.assert_series_equal(weights, min_es_weights(scen))

    def test_walk_forward_tests_exactly_this_model(self):
        prices = gbm_prices(n=120)
        results = walk_forward(prices, pd.Series({"A": 0.5, "B": 0.5}), days=5, window=60, n=300, seed=2)
        i = 60 + 2 * 5  # third rebalance
        weights, _ = model_weights(log_returns(prices.iloc[i - 60:i + 1]), 300, 5, seed=2 + i)
        np.testing.assert_allclose(results.iloc[2][["A", "B"]].astype(float), weights)


if __name__ == "__main__":
    unittest.main()
