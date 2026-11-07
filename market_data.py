"""Download and validate independent historical price snapshots for Monte Carlo.

Adapted from berkeksioglu06/quant-ml market_data.py at dd220ad.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["ZETA", "RKLB", "PLTR", "VLN", "HIMS", "MRVL", "ACHR", "EOSE"]
OPTIONS = dict(interval="1d", auto_adjust=False, back_adjust=False,
               repair=False, actions=True, keepna=True)
RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
          "Adj Close": "adj_close", "Volume": "volume", "Dividends": "dividends",
          "Stock Splits": "splits", "Capital Gains": "capital_gains"}


def calendar(start, end):
    # Pad bounds so weekend/holiday queries remain inside the session range.
    return xc.get_calendar("XNYS", start=pd.Timestamp(start) - pd.Timedelta(days=14),
                           end=pd.Timestamp(end) + pd.Timedelta(days=14))


def last_complete_session(now: datetime) -> pd.Timestamp:
    now = pd.Timestamp(now).tz_convert("UTC")
    today = now.tz_localize(None).normalize()
    cal = calendar(today - pd.Timedelta(days=20), today + pd.Timedelta(days=1))
    for day in reversed(cal.sessions_in_range(today - pd.Timedelta(days=14), today)):
        if now >= cal.session_close(day) + pd.Timedelta(minutes=15):
            return pd.Timestamp(day).tz_localize(None)
    raise ValueError("No completed session in the last 14 days")


def clean_prices(raw: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Empty provider response")
    frame = raw.copy().rename(columns=RENAME)
    dates = pd.DatetimeIndex(frame.index)
    if dates.tz is not None:
        dates = dates.tz_convert("America/New_York").tz_localize(None)
    frame.index = dates.normalize()
    frame.index.name = "date"
    frame = frame.loc[frame.index <= cutoff].reset_index().drop_duplicates()
    if frame["date"].duplicated().any():
        raise ValueError("Conflicting duplicate dates")
    frame = frame.set_index("date").sort_index()
    if frame.empty:
        raise ValueError("No completed daily bars")
    required = ["open", "high", "low", "close", "adj_close", "volume"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing OHLCV or adjusted-close columns")
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite OHLCV or adjusted close")
    if (frame[required[:-1]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Non-positive prices or negative volume")
    tolerance = frame["close"].abs() * 1e-6
    if ((frame["high"] + tolerance < frame[["open", "close", "low"]].max(axis=1)) |
        (frame["low"] - tolerance > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC relationships")
    return frame


def align_prices(frames: dict[str, pd.DataFrame], cutoff: pd.Timestamp):
    start = max(frame.index.min() for frame in frames.values())
    cal = calendar(start - pd.Timedelta(days=7), cutoff + pd.Timedelta(days=1))
    sessions = cal.sessions_in_range(start, cutoff).tz_localize(None)
    if len(sessions) < 2:
        raise ValueError("Fewer than two common sessions")
    for symbol, frame in frames.items():
        missing = sessions.difference(frame.index)
        extra = frame.index.difference(cal.sessions)
        # Check extra dates only within the common calendar's bounds.
        extra = extra[(extra >= start) & (extra <= cutoff)]
        if len(missing) or len(extra):
            raise ValueError(f"{symbol}: {len(missing)} missing common sessions, "
                             f"{len(extra)} non-session dates")
    return {
        field: pd.DataFrame({s: f[field].reindex(sessions) for s, f in frames.items()})
        .rename_axis("date") for field in ("adj_close", "close")
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(symbols: list[str], start: str, output: Path) -> Path:
    now = datetime.now(UTC)
    cutoff = last_complete_session(now)
    if pd.Timestamp(start) >= cutoff:
        raise ValueError("Start date must precede the latest completed session")
    snap = output / f"yahoo_{now:%Y%m%dT%H%M%S%fZ}"
    (snap / "raw").mkdir(parents=True, exist_ok=False)
    (snap / "clean").mkdir()
    manifest = {"source": "Yahoo Finance via yfinance", "downloaded_at": now.isoformat(),
                "start": start, "last_complete_session": str(cutoff.date()),
                "options": OPTIONS, "calendar": "XNYS (US equity sessions)",
                "versions": {p: version(p) for p in
                             ("yfinance", "pandas", "numpy", "exchange-calendars")},
                "symbols": {}, "status": "RUNNING"}
    frames, issues, coverage = {}, [], []
    for symbol in symbols:
        print(f"Downloading {symbol}...", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(start=start, end=str((cutoff + pd.Timedelta(days=1)).date()),
                                 **OPTIONS)
            path = snap / "raw" / f"{symbol}.csv"
            raw.to_csv(path, index_label="date")
            record = {"raw_sha256": digest(path), "provider_rows": len(raw)}
            manifest["symbols"][symbol] = record
            frame = clean_prices(raw, cutoff)
            metadata = ticker.get_history_metadata()
            record["currency"] = metadata.get("currency")
            if record["currency"] != "USD":
                raise ValueError(f"Expected USD currency, received {record['currency']!r}")
            clean_path = snap / "clean" / f"{symbol}.csv"
            frame.to_csv(clean_path)
            record.update(clean_sha256=digest(clean_path), rows=len(frame),
                          first=str(frame.index.min().date()), last=str(frame.index.max().date()))
            coverage.append({"symbol": symbol, "rows": len(frame),
                             "first": record["first"], "last": record["last"]})
            if frame.index.min() > pd.Timestamp(start) + pd.Timedelta(days=7):
                issues.append({"symbol": symbol, "severity": "WARNING", "code": "SHORT_HISTORY",
                               "message": f"Requested {start}; available from {record['first']}"})
            if frame.index.max() != cutoff:
                raise ValueError(f"Stale series: {frame.index.max().date()} vs {cutoff.date()}")
            moves = frame["adj_close"].pct_change(fill_method=None).abs() > 0.25
            for day in frame.index[moves]:
                issues.append({"symbol": symbol, "severity": "WARNING", "code": "LARGE_MOVE",
                               "message": f"{day.date()}: adjusted-close move exceeds 25%; review"})
            for day in frame.index[frame.get("splits", pd.Series(0, index=frame.index)) > 0]:
                issues.append({"symbol": symbol, "severity": "WARNING", "code": "SPLIT",
                               "message": f"{day.date()}: verify provider adjustment"})
            frames[symbol] = frame
        except Exception as error:
            manifest["symbols"].setdefault(symbol, {})["error"] = str(error)
            issues.append({"symbol": symbol, "severity": "ERROR", "code": "SYMBOL_FAILED",
                           "message": str(error)})
        time.sleep(0.3)
    if len(frames) == len(symbols):
        try:
            aligned = align_prices(frames, cutoff)
            if len(aligned["close"]) < 252:
                raise ValueError("Fewer than 252 common sessions; insufficient calibration history")
            aligned["adj_close"].to_csv(snap / "prices.csv")
            aligned["close"].to_csv(snap / "close_prices.csv")
            manifest["common"] = {"first": str(aligned["close"].index.min().date()),
                                  "last": str(cutoff.date()), "rows": len(aligned["close"])}
        except ValueError as error:
            issues.append({"symbol": "ALL", "severity": "ERROR", "code": "ALIGNMENT",
                           "message": str(error)})
    manifest["status"] = "FAILED" if any(i["severity"] == "ERROR" for i in issues) else "READY"
    pd.DataFrame(issues, columns=["symbol", "severity", "code", "message"]).to_csv(
        snap / "issues.csv", index=False)
    pd.DataFrame(coverage, columns=["symbol", "rows", "first", "last"]).to_csv(
        snap / "coverage.csv", index=False)
    manifest["files"] = {p.relative_to(snap).as_posix(): digest(p)
                         for p in snap.rglob("*.csv")}
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{manifest['status']}: {snap}")
    for issue in issues:
        if issue["severity"] == "ERROR":
            print(f"  {issue['symbol']}: {issue['message']}")
    if manifest["status"] == "FAILED":
        raise RuntimeError(f"Snapshot incomplete; inspect {snap / 'issues.csv'}")
    return snap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]
    if len(set(symbols)) != len(symbols) or any(not re.fullmatch(r"[A-Z][A-Z0-9.-]*", s) for s in symbols):
        parser.error("Symbols must be unique ticker names")
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.start):
            raise ValueError("Start date must use YYYY-MM-DD")
        download(symbols, args.start, args.output)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
