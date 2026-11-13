import unittest

import numpy as np
import pandas as pd

from frontier import (against_current, compare, current_weights, frontier, measure,
                      min_es_weights, random_mixes, scenarios)
from risk import gbm_var_es, var_es


def daily_returns(n=500, seed=3):
    # Three stocks: calm, wild, and a wild one that moves with the second.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    z[:, 2] = 0.8 * z[:, 1] + 0.6 * z[:, 2]
    return pd.DataFrame(z * [0.01, 0.03, 0.04], columns=["A", "B", "C"])


def weights(*values):
    return pd.Series(values, index=["A", "B", "C"])


class ScenarioTests(unittest.TestCase):
    def test_same_seed_gives_the_same_scenarios(self):
        returns = daily_returns()
        for source in ("gbm", "bootstrap"):
            first = scenarios(returns, 200, 5, seed=1, source=source)
            pd.testing.assert_frame_equal(first, scenarios(returns, 200, 5, seed=1, source=source))
            self.assertEqual(first.shape, (200, 3))
            self.assertFalse(first.equals(scenarios(returns, 200, 5, seed=2, source=source)))

    def test_gbm_single_stock_matches_the_exact_es(self):
        returns = daily_returns()[["B"]]
        scen = scenarios(returns, 200_000, 21)
        exact = gbm_var_es(1.0, 0.0, returns["B"].std(), 21, 0.95)[1]
        self.assertAlmostEqual(measure(scen, pd.Series({"B": 1.0}))["es"], exact, delta=0.003)

    def test_zero_drift_centers_log_returns_in_both_sources(self):
        returns = daily_returns() + 0.002
        for source in ("gbm", "bootstrap"):
            zero = np.log1p(scenarios(returns, 50_000, 10, source=source, block=1)).mean()
            kept = np.log1p(scenarios(returns, 50_000, 10, drift="historical", source=source,
                                      block=1)).mean()
            np.testing.assert_allclose(zero, 0, atol=0.003)
            np.testing.assert_allclose(kept, returns.mean() * 10, atol=0.003)

    def test_bootstrap_keeps_whole_days_together(self):
        # B is exactly twice A on every real day, so it must be in every resampled scenario too.
        returns = daily_returns()[["A"]].assign(B=lambda frame: 2 * frame["A"])
        scen = np.log1p(scenarios(returns, 300, 21, drift="historical", source="bootstrap"))
        np.testing.assert_allclose(scen["B"], 2 * scen["A"])

    def test_bootstrap_blocks_are_consecutive_days(self):
        # Daily log return = its row number, so a 4-day block starting at row i sums to 4i + 6.
        returns = pd.DataFrame({"A": np.arange(100.0), "B": np.arange(100.0)}) * 1e-4
        scen = np.log1p(scenarios(returns, 500, 4, drift="historical", source="bootstrap", block=4))
        starts = (scen["A"] / 1e-4 - 6) / 4
        np.testing.assert_allclose(starts, starts.round(), atol=1e-6)
        self.assertTrue(starts.round().between(0, 96).all())

    def test_bad_inputs_are_rejected(self):
        returns = daily_returns()
        with self.assertRaisesRegex(ValueError, "Unknown source"):
            scenarios(returns, source="history")
        with self.assertRaisesRegex(ValueError, "Block"):
            scenarios(returns, source="bootstrap", block=0)
        with self.assertRaisesRegex(ValueError, "at least"):
            scenarios(returns.iloc[:20])


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scen = scenarios(daily_returns(), 2_000, 21)

    def test_measure_is_the_weighted_sum_of_stock_returns(self):
        w = weights(0.5, 0.3, 0.2)
        gain = self.scen.to_numpy() @ w.to_numpy()
        result = measure(self.scen, w)
        self.assertAlmostEqual(result["mean"], gain.mean())
        self.assertAlmostEqual(result["es"], var_es(-gain, 0.95)[1])
        self.assertAlmostEqual(result["upside"], np.sort(gain)[-100])  # mildest of the best 5%
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            measure(self.scen, weights(0.5, 0.3, 0.3))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            measure(self.scen, weights(1.2, -0.2, 0.0))

    def test_weights_are_long_only_and_sum_to_one(self):
        w = min_es_weights(self.scen)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w >= 0).all())
        self.assertEqual(list(w.index), ["A", "B", "C"])

    def test_no_other_weights_have_a_lower_es(self):
        best = measure(self.scen, min_es_weights(self.scen))["es"]
        for w in np.random.default_rng(0).dirichlet(np.ones(3), 300):
            self.assertGreaterEqual(measure(self.scen, weights(*w))["es"], best - 1e-9)

    def test_identical_stocks_cannot_diversify_but_independent_ones_can(self):
        rng = np.random.default_rng(1)
        one = rng.normal(0, 0.1, 4_000)
        same = pd.DataFrame({"A": one, "B": one})
        self.assertAlmostEqual(measure(same, min_es_weights(same))["es"],
                               measure(same, pd.Series({"A": 1.0, "B": 0.0}))["es"])
        apart = pd.DataFrame({"A": one, "B": rng.normal(0, 0.1, 4_000)})
        w = min_es_weights(apart)
        self.assertAlmostEqual(w["A"], 0.5, delta=0.05)
        self.assertLess(measure(apart, w)["es"], 0.8 * measure(apart, pd.Series({"A": 1.0, "B": 0.0}))["es"])

    def test_target_is_met_and_costs_risk(self):
        low = measure(self.scen, min_es_weights(self.scen))
        target = (low["mean"] + self.scen.mean().max()) / 2
        reached = measure(self.scen, min_es_weights(self.scen, target))
        self.assertGreaterEqual(reached["mean"], target - 1e-9)
        self.assertGreater(reached["es"], low["es"])
        with self.assertRaisesRegex(ValueError, "No portfolio reaches"):
            min_es_weights(self.scen, self.scen.mean().max() + 0.01)

    def test_frontier_runs_from_safest_mix_to_best_single_stock(self):
        curve = frontier(self.scen, points=6)
        self.assertEqual(len(curve), 6)
        self.assertTrue(curve["mean"].is_monotonic_increasing)
        self.assertTrue(curve["es"].is_monotonic_increasing)
        np.testing.assert_allclose(curve[["A", "B", "C"]].sum(axis=1), 1)
        self.assertAlmostEqual(curve["es"].iat[0], measure(self.scen, min_es_weights(self.scen))["es"])
        self.assertEqual(curve[["A", "B", "C"]].iloc[-1].max(), 1)  # everything in one stock
        self.assertAlmostEqual(curve["mean"].iat[-1], self.scen.mean().max())

    def test_random_mixes_never_beat_the_frontier(self):
        curve, cloud = frontier(self.scen, points=6), random_mixes(self.scen, 200)
        self.assertGreaterEqual(cloud["es"].min(), curve["es"].iat[0] - 1e-9)
        for mix in cloud.itertuples():  # no random mix has both more mean and less ES
            for point in curve[curve["mean"] <= mix.mean].itertuples():
                self.assertGreaterEqual(mix.es, point.es - 1e-9)


class CompareTests(unittest.TestCase):
    def test_current_weights_are_value_shares(self):
        symbols = pd.Index(["A", "B", "C"])
        w = current_weights(symbols, pd.Series({"A": 10.0, "B": 20.0, "C": 5.0}),
                            pd.Series({"B": 1.0, "A": 2.0}))  # aligned by symbol; C is not held
        self.assertEqual(w.to_dict(), {"A": 0.5, "B": 0.5, "C": 0.0})

    def test_compare_improves_on_a_concentrated_portfolio(self):
        scen = scenarios(daily_returns(), 2_000, 21)
        table = against_current(compare(scen, weights(0.0, 0.1, 0.9)))
        self.assertEqual(list(table.index), ["current", "equal weight", "min ES at same mean", "min ES"])
        self.assertEqual(table.at["current", "es_cut"], 0)
        self.assertEqual(table.at["current", "upside_kept"], 1)
        matched = table.loc["min ES at same mean"]
        self.assertGreaterEqual(matched["mean"], table.at["current", "mean"] - 1e-9)
        self.assertGreater(matched["es_cut"], 0)
        self.assertGreaterEqual(table.at["min ES", "es_cut"], matched["es_cut"] - 1e-9)
        np.testing.assert_allclose(table[["A", "B", "C"]].sum(axis=1).astype(float), 1)


if __name__ == "__main__":
    unittest.main()
