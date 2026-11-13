import unittest

import numpy as np
import pandas as pd

from test_backtest import gbm_prices
from walkforward import STRATEGIES, summarize, walk_forward

CURRENT = pd.Series({"A": 0.2, "B": 0.8})


class WalkForwardTests(unittest.TestCase):
    def test_weights_ignore_prices_after_their_date(self):
        prices = gbm_prices(n=200)
        base = walk_forward(prices, CURRENT, days=5, window=60, n=300)
        changed = prices.copy()
        t = base.index[3]
        changed.loc[changed.index > t] *= 0.5  # crash right after t
        moved = walk_forward(changed, CURRENT, days=5, window=60, n=300)
        pd.testing.assert_series_equal(base.loc[t, ["A", "B"]], moved.loc[t, ["A", "B"]])
        self.assertAlmostEqual(moved.loc[t, "current"], (1 + base.loc[t, "current"]) / 2 - 1)

    def test_realized_return_is_buy_and_hold_between_the_two_closes(self):
        prices = gbm_prices(n=120)
        results = walk_forward(prices, CURRENT, days=5, window=60, n=300)
        t = results.index[0]
        self.assertEqual(t, prices.index[60])  # first date with a full window
        end = prices.index[prices.index.get_loc(t) + 5]
        self.assertEqual(results.loc[t, "end"], end)
        moves = prices.loc[end] / prices.loc[t] - 1
        self.assertAlmostEqual(results.loc[t, "current"], CURRENT @ moves)
        self.assertAlmostEqual(results.loc[t, "equal weight"], moves.mean())
        self.assertAlmostEqual(results.loc[t, "min ES"], results.loc[t, ["A", "B"]] @ moves)

    def test_holding_periods_do_not_overlap_by_default(self):
        results = walk_forward(gbm_prices(n=150), CURRENT, days=7, window=60, n=200)
        self.assertTrue((results.index[1:] >= results["end"].iloc[:-1]).all())
        overlapping = walk_forward(gbm_prices(n=150), CURRENT, days=7, window=60, n=200, step=1)
        self.assertGreater(len(overlapping), 6 * len(results))

    def test_min_es_cuts_realized_tail_losses_when_the_model_is_true(self):
        # B is three times as volatile as A, and the current mix sits mostly in B.
        prices = gbm_prices(n=2500, vols=(0.01, 0.03))
        table = summarize(walk_forward(prices, CURRENT, days=5, window=120, n=500))
        self.assertEqual(list(table.index), list(STRATEGIES))
        self.assertGreater(table.at["min ES", "es_cut"], 0.4)
        self.assertLess(table.at["min ES", "upside_kept"], 1)
        self.assertEqual(table.at["current", "es_cut"], 0)

    def test_summary_measures_the_realized_tails(self):
        gains = np.linspace(-0.20, 0.19, 40)  # 5% of 40 periods = the worst and best two
        results = pd.DataFrame({name: gains for name in STRATEGIES})
        results["min ES"] = gains / 2
        table = summarize(results)
        self.assertAlmostEqual(table.at["current", "es"], 0.195)
        self.assertAlmostEqual(table.at["current", "upside"], 0.18)
        self.assertAlmostEqual(table.at["current", "worst"], -0.20)
        self.assertAlmostEqual(table.at["min ES", "es_cut"], 0.5)
        self.assertAlmostEqual(table.at["min ES", "upside_kept"], 0.5)

    def test_bad_inputs_are_rejected(self):
        prices = gbm_prices(n=120)
        with self.assertRaisesRegex(ValueError, "Window"):
            walk_forward(prices, CURRENT, window=10)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            walk_forward(prices, CURRENT * 2, days=5, window=60)
        with self.assertRaisesRegex(ValueError, "Need more than"):
            walk_forward(prices.iloc[:60], CURRENT, days=5, window=60)


if __name__ == "__main__":
    unittest.main()
