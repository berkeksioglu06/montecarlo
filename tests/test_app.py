import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from app import Handler, load_history, optimization_report, read_request, simulation_report
from test_backtest import SHARES, gbm_prices


class RequestTests(unittest.TestCase):
    def test_positions_and_horizon_are_read(self):
        shares, days = read_request({"positions": [{"symbol": " pltr ", "shares": "2.5"},
                                                   {"symbol": "BRK.B", "shares": 1}], "days": 5})
        self.assertEqual(shares.to_dict(), {"PLTR": 2.5, "BRK.B": 1.0})
        self.assertEqual(days, 5)
        self.assertEqual(read_request({"positions": [{"symbol": "A", "shares": 1}]})[1], 21)

    def test_bad_requests_are_rejected(self):
        for payload, message in (
                ({}, "between 1 and"),
                ({"positions": [{"symbol": "$$$", "shares": 1}]}, "Not a ticker"),
                ({"positions": [{"symbol": "A", "shares": 1}, {"symbol": "a", "shares": 1}]}, "twice"),
                ({"positions": [{"symbol": "A", "shares": "many"}]}, "positive number"),
                ({"positions": [{"symbol": "A", "shares": 0}]}, "positive number"),
                ({"positions": [{"symbol": "A", "shares": 1}], "days": 0}, "Horizon")):
            with self.assertRaisesRegex(ValueError, message):
                read_request(payload)


class ReportTests(unittest.TestCase):
    def test_simulation_report_describes_the_simulated_paths(self):
        prices = gbm_prices(n=300)
        report = simulation_report(prices, SHARES, days=5, n_paths=2_000)
        json.dumps(report)  # everything must be plain JSON
        self.assertAlmostEqual(report["value"], prices.iloc[-1] @ SHARES, places=2)
        self.assertEqual(report["as_of"], str(prices.index[-1].date()))
        self.assertEqual(report["returns"], 252)  # only the tested window is used
        bands = report["bands"]
        self.assertEqual(len(bands["p50"]), 6)  # today plus five days
        self.assertTrue(bands["p5"][-1] < bands["p50"][-1] < bands["p95"][-1])
        self.assertAlmostEqual(sum(report["histogram"]["share"]), 1, places=3)
        self.assertGreater(report["risk"][0]["es"], report["risk"][0]["var"])
        self.assertGreater(report["risk"][1]["var"], report["risk"][0]["var"])
        self.assertAlmostEqual(sum(row["weight"] for row in report["positions"]), 1, places=3)
        self.assertAlmostEqual(sum(row["risk_share"] for row in report["positions"]), 1, places=3)

    def test_optimization_report_compares_the_model_with_the_portfolio(self):
        prices = gbm_prices(n=300, vols=(0.01, 0.03))
        shares = pd.Series({"A": 1.0, "B": 9.0})  # mostly in the volatile stock
        report = optimization_report(prices, shares, days=5, n=1_000)
        json.dumps(report)
        current, model = report["portfolios"]["current"], report["portfolios"]["model"]
        self.assertLess(model["es"], current["es"])
        self.assertGreater(model["es_cut"], 0.3)
        rows = {row["symbol"]: row for row in report["positions"]}
        self.assertAlmostEqual(sum(row["model"] for row in rows.values()), 1, places=3)
        self.assertGreater(rows["A"]["model"], rows["A"]["current"])
        # The model's share counts are worth what the portfolio is worth today.
        value = sum(row["model_shares"] * prices[symbol].iloc[-1] for symbol, row in rows.items())
        self.assertAlmostEqual(value, prices.iloc[-1] @ shares, delta=0.01 * value)
        self.assertEqual(len(report["frontier"]), 10)

    def test_history_comes_from_a_saved_snapshot_when_one_has_the_symbols(self):
        prices = load_history(["PLTR", "MRVL"])  # both are in the repository's snapshots
        self.assertEqual(list(prices.columns), ["PLTR", "MRVL"])
        self.assertGreater(len(prices), 252)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.address = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path, payload):
        request = Request(self.address + path, json.dumps(payload).encode(),
                          {"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.loads(response.read())

    def test_page_and_default_portfolio_are_served(self):
        with urlopen(self.address + "/") as response:
            self.assertIn(b"<title>Monte Carlo</title>", response.read())
        with urlopen(self.address + "/api/portfolio") as response:
            self.assertTrue(all({"symbol", "shares"} <= set(row) for row in json.loads(response.read())))

    def test_simulation_runs_over_http(self):
        report = self.post("/api/simulate", {"positions": [{"symbol": "PLTR", "shares": 2},
                                                           {"symbol": "MRVL", "shares": 3}], "days": 5})
        self.assertEqual(report["days"], 5)
        self.assertEqual([row["symbol"] for row in report["positions"]], ["PLTR", "MRVL"])

    def test_bad_input_is_an_error_message_not_a_crash(self):
        for path, payload in (("/api/simulate", {"positions": []}), ("/api/nope", {})):
            with self.assertRaises(HTTPError) as caught:
                self.post(path, payload)
            self.assertIn(caught.exception.code, (400, 404))
            self.assertIn("error", json.loads(caught.exception.read()))


if __name__ == "__main__":
    unittest.main()
