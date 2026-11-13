"""Local web app: simulate a portfolio, then reweight it. Run it and use the page it opens."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from frontier import against_current, current_weights, frontier, measure, random_mixes
from market_data import download
from returns import DATA, load_prices, log_returns
from risk import LEVELS, losses, max_drawdown, var_es, variance_contribution
from simulation import PORTFOLIO, load_portfolio, simulate
from weights import model_weights

ROOT = Path(__file__).parent
PAGE = ROOT / "web" / "index.html"
APP_DATA = DATA / "app"  # snapshots the app downloads: ignored by Git and never picked up by the scripts
WINDOW = 252  # calibration returns, the window that backtest.py and walkforward.py test
HISTORY_START = "2021-01-01"
MAX_POSITIONS = 30


def read_request(payload: dict) -> tuple[pd.Series, int]:
    """Shares by symbol and the horizon in trading days, checked before anything is computed."""
    shares = {}
    for row in payload.get("positions") or []:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol):
            raise ValueError(f"Not a ticker: {symbol or 'blank'}")
        if symbol in shares:
            raise ValueError(f"{symbol} is listed twice")
        try:
            count = float(row.get("shares"))
        except (TypeError, ValueError):
            count = float("nan")
        if not np.isfinite(count) or count <= 0:
            raise ValueError(f"{symbol}: shares must be a positive number")
        shares[symbol] = count
    if not 1 <= len(shares) <= MAX_POSITIONS:
        raise ValueError(f"Enter between 1 and {MAX_POSITIONS} positions")
    days = payload.get("days", 21)
    if not isinstance(days, int) or not 1 <= days <= 126:
        raise ValueError("Horizon must be between 1 and 126 trading days")
    return pd.Series(shares), days


def ready_snapshots() -> list[Path]:
    """READY snapshots, newest first: the app's own downloads and the repository's."""
    found = [path.parent for folder in (APP_DATA, DATA) for path in folder.glob("yahoo_*/manifest.json")
             if json.loads(path.read_text(encoding="utf-8")).get("status") == "READY"]
    return sorted(found, key=lambda snapshot: snapshot.name, reverse=True)


def load_history(symbols: list[str], fresh: bool = False) -> pd.DataFrame:
    """Adjusted closes of the symbols: the newest snapshot that has them all, else a new download.

    `fresh` skips the saved snapshots and downloads up to the latest completed session.
    """
    for snapshot in [] if fresh else ready_snapshots():
        if set(symbols) <= set(pd.read_csv(snapshot / "prices.csv", nrows=0).columns):
            return load_prices(snapshot)[symbols]
    try:
        return load_prices(download(symbols, HISTORY_START, APP_DATA))[symbols]
    except RuntimeError:
        # A failed download leaves its reasons in issues.csv; report them and drop the folder.
        failed = max(APP_DATA.glob("yahoo_*"), key=lambda snapshot: snapshot.name)
        issues = pd.read_csv(failed / "issues.csv")
        reasons = [f"{row.symbol}: {row.message}" for row in issues.itertuples() if row.severity == "ERROR"]
        shutil.rmtree(failed)
        raise ValueError("Could not get prices. " + "; ".join(reasons)) from None


def rounded(values, digits: int = 5) -> list:
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def simulation_report(prices: pd.DataFrame, shares: pd.Series, days: int = 21,
                      n_paths: int = 10_000, seed: int = 0) -> dict:
    """Everything the simulation page shows, from the same simulate() the scripts use."""
    known = prices.iloc[-(WINDOW + 1):]
    returns, start = log_returns(known), known.iloc[-1]
    values = simulate(returns, start, shares, n_paths, days, seed)
    now, loss = float(values[0, 0]), losses(values)
    shown = np.random.default_rng(seed).choice(n_paths, min(40, n_paths), replace=False)
    change = -loss / now
    low, high = np.percentile(change, [0.5, 99.5])  # the end bars collect the few paths beyond
    counts, edges = np.histogram(np.clip(change, low, high), bins=40)
    drawdown = max_drawdown(values)
    split = variance_contribution(returns, start, shares)
    return {
        "as_of": str(known.index[-1].date()), "returns": len(returns), "paths": n_paths, "days": days,
        "value": round(now, 2),
        "bands": {f"p{q}": rounded(np.percentile(values, q, axis=0), 2) for q in (5, 25, 50, 75, 95)},
        "sample": rounded(values[shown], 2),
        "histogram": {"edges": rounded(edges), "share": rounded(counts / n_paths)},
        "risk": [dict(zip(("level", "var", "es"), rounded([level, *np.divide(var_es(loss, level), now)])))
                 for level in LEVELS],
        "upside": round(float(var_es(-loss, 0.95)[0]) / now, 5),
        "drawdown": rounded([np.median(drawdown), np.percentile(drawdown, 95)]),
        "positions": [{"symbol": symbol, "price": round(float(start[symbol]), 2),
                       "weight": round(float(row.weight), 5),
                       "risk_share": round(float(row.variance_share), 5)}
                      for symbol, row in split.iterrows()],
    }


def optimization_report(prices: pd.DataFrame, shares: pd.Series, days: int = 21,
                        n: int = 5_000, seed: int = 0) -> dict:
    """The weights model against the portfolio as it is, both measured on the same scenarios."""
    known = prices.iloc[-(WINDOW + 1):]
    returns, start = log_returns(known), known.iloc[-1]
    model, scen = model_weights(returns, n, days, seed)
    current = current_weights(returns.columns, start, shares)
    equal = pd.Series(1 / len(current), index=current.index)
    table = against_current(pd.DataFrame({"current": measure(scen, current), "model": measure(scen, model),
                                          "equal": measure(scen, equal)}).T)
    value = float(start @ shares)
    return {
        "as_of": str(known.index[-1].date()), "days": days,
        "portfolios": {name: {key: round(float(number), 5) for key, number in row.items()}
                       for name, row in table.iterrows()},
        "positions": [{"symbol": symbol, "current": round(float(current[symbol]), 5),
                       "model": round(float(model[symbol]), 5), "shares": float(shares[symbol]),
                       "model_shares": round(value * float(model[symbol]) / float(start[symbol]), 2)}
                      for symbol in returns.columns],
        "frontier": frontier(scen, 10)[["es", "upside"]].round(5).to_dict("records"),
        "cloud": random_mixes(scen, 300, seed)[["es", "upside"]].round(5).to_dict("records"),
    }


def default_positions() -> list[dict]:
    """The user's portfolio.csv if it is there and valid, else the example."""
    for path in (PORTFOLIO, ROOT / "portfolio.example.csv"):
        try:
            return [{"symbol": symbol, "shares": float(count)}
                    for symbol, count in load_portfolio(path)["shares"].items()]
        except (ValueError, FileNotFoundError):
            continue
    return []


class Handler(BaseHTTPRequestHandler):
    REPORTS = {"/api/simulate": simulation_report, "/api/optimize": optimization_report}

    def send(self, status: int, body: bytes, kind: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/portfolio":
            self.send(200, json.dumps(default_positions()).encode())
        else:
            self.send(404, b'{"error": "Not found"}')

    def do_POST(self):
        report = self.REPORTS.get(self.path)
        if report is None:
            return self.send(404, b'{"error": "Not found"}')
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            payload = json.loads(body or b"{}")
            shares, days = read_request(payload)
            prices = load_history(list(shares.index), fresh=payload.get("refresh") is True)
            status, result = 200, report(prices, shares, days)
        except (ValueError, FileNotFoundError) as error:
            status, result = 400, {"error": str(error)}
        except Exception:  # keep the server up and tell the page; the details go to the terminal
            traceback.print_exc()
            status, result = 500, {"error": "Something went wrong; the terminal has the details"}
        self.send(status, json.dumps(result).encode())

    def log_message(self, *args):  # the terminal stays quiet apart from downloads and errors
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page")
    args = parser.parse_args()
    address = f"http://127.0.0.1:{args.port}"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)  # this machine only
    except OSError as error:
        parser.exit(1, f"Error: port {args.port} is not free ({error}); try --port\n")
    print(f"Monte Carlo is running at {address}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
