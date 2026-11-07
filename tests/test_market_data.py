import unittest
from datetime import UTC, datetime

import pandas as pd

from market_data import align_prices, clean_prices, last_complete_session


def bars(dates):
    return pd.DataFrame({"Open": 10., "High": 11., "Low": 9., "Close": 10.,
                         "Adj Close": 10., "Volume": 100}, index=pd.to_datetime(dates))


class MarketDataTests(unittest.TestCase):
    def test_incomplete_session_and_exact_duplicates(self):
        raw = bars(["2026-11-04", "2026-11-04", "2026-11-05"])
        result = clean_prices(raw, pd.Timestamp("2026-11-04"))
        self.assertEqual(len(result), 1)

    def test_conflicting_duplicate_is_rejected(self):
        raw = bars(["2026-11-04", "2026-11-04"])
        raw.iloc[1, raw.columns.get_loc("Close")] = 10.5
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            clean_prices(raw, pd.Timestamp("2026-11-04"))

    def test_missing_session_is_not_silently_filled(self):
        complete = clean_prices(bars(["2026-11-02", "2026-11-03", "2026-11-04"]),
                                pd.Timestamp("2026-11-04"))
        missing = complete.drop(pd.Timestamp("2026-11-03"))
        with self.assertRaisesRegex(ValueError, "missing common sessions"):
            align_prices({"AAA": complete, "BBB": missing}, pd.Timestamp("2026-11-04"))

    def test_early_close_buffer(self):
        # Black Friday closes at 18:00 UTC, with a 15 minute publication buffer.
        self.assertEqual(last_complete_session(datetime(2025, 11, 28, 18, 10, tzinfo=UTC)),
                         pd.Timestamp("2025-11-26"))
        self.assertEqual(last_complete_session(datetime(2025, 11, 28, 18, 20, tzinfo=UTC)),
                         pd.Timestamp("2025-11-28"))

    def test_weekend_uses_friday(self):
        self.assertEqual(last_complete_session(datetime(2026, 11, 7, 1, tzinfo=UTC)),
                         pd.Timestamp("2026-11-06"))


if __name__ == "__main__":
    unittest.main()
