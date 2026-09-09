"""Continuous market data collector — the eye that never sleeps.

Runs INDEPENDENTLY of trading (owner requirement): even with the paper
trader stopped, candles keep flowing into data/raw so the model's history
never develops gaps.

Safety properties:
  - file lock (flock): two collectors can never write the same CSV
  - gap self-heal: on start (and every cycle) it walks each symbol's CSV
    and fetches only the MISSING hours between last stored candle and now
  - appends only (never rewrites history)
  - rate-limit friendly (ccxt enableRateLimit)
  - network failures are logged and retried next cycle — NEVER crash
"""
import argparse
import fcntl
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd


def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
          flush=True)


def load_symbols(raw_dir, extra=None):
    """Symbols = every CSV already in raw_dir (training universe) + extras."""
    syms = []
    if os.path.isdir(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            if f.endswith(".csv"):
                syms.append(f[:-4].replace("_", "/"))
    for s in extra or []:
        s = s.upper()
        if s not in syms:
            syms.append(s)
    return syms


def fetch_gap(exchange, symbol, timeframe, raw_path, since_default):
    """Fetch only candles missing after the last stored timestamp.
    Returns the new DataFrame (possibly empty)."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    if os.path.exists(raw_path):
        last = pd.read_csv(raw_path, usecols=["datetime"])
        since = int(pd.Timestamp(last.iloc[-1, 0]) .timestamp() * 1000) + tf_ms
    else:
        since = since_default
    rows = []
    cursor = since
    now_ms = int(time.time() * 1000)
    while cursor < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                        since=cursor, limit=1000)
        except Exception as e:
            _log(f"  {symbol}: fetch error at {cursor}: {e} — will retry next cycle")
            break
        if not batch:
            break
        # drop the still-open candle: only CLOSED candles enter history
        closed = [r for r in batch if r[0] + tf_ms <= now_ms]
        rows.extend(closed)
        last_ts = batch[-1][0]
        nxt = last_ts + tf_ms
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 1000 and not closed:
            break
        if not closed:
            # batch had only the open candle — nothing new yet
            break
        time.sleep(exchange.rateLimit / 1000)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                     "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def append_csv(raw_path, new_df):
    """Append new candles to the CSV (dedup on write, keep order)."""
    new_df = new_df.reset_index(drop=True)
    if os.path.exists(raw_path):
        old = pd.read_csv(raw_path, parse_dates=["datetime"])
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates("datetime").sort_values("datetime")
        combined.to_csv(raw_path, index=False)
        return len(combined) - len(old)
    new_df.to_csv(raw_path, index=False)
    return len(new_df)


def run_cycle(exchange, raw_dir, symbols, timeframe, since_ms):
    added_total = 0
    for sym in symbols:
        path = os.path.join(raw_dir, sym.replace("/", "_") + ".csv")
        try:
            new = fetch_gap(exchange, sym, timeframe, path, since_ms)
            if new is not None and len(new):
                added = append_csv(path, new)
                added_total += added
                _log(f"  {sym}: +{added} candles (last {new.datetime.iloc[-1]})")
            else:
                _log(f"  {sym}: up to date")
        except Exception as e:
            _log(f"  {sym}: FAILED ({e}) — skipped this cycle, will retry")
    return added_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=[],
                   help="extra symbols beyond the CSVs already in raw dir")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--since", default="2020-01-01",
                   help="start date for brand-new symbols")
    p.add_argument("--interval-min", type=float, default=5.0,
                   help="minutes between cycles")
    a = p.parse_args()

    import ccxt
    os.makedirs(a.raw_dir, exist_ok=True)
    lock_path = os.path.join(a.raw_dir, ".collector.lock")
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log("another collector is already running — exiting (this is safe)")
        sys.exit(0)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot", "fetchMarkets": ["spot"]},
    })
    exchange.load_markets()
    since_ms = int(pd.Timestamp(a.since, tz="UTC").timestamp() * 1000)

    _log(f"collector started: {a.interval_min}min cycle, {a.timeframe} candles")
    while True:
        symbols = load_symbols(a.raw_dir, a.symbols)
        _log(f"cycle: {len(symbols)} symbols")
        try:
            run_cycle(exchange, a.raw_dir, symbols, a.timeframe, since_ms)
        except Exception as e:
            _log(f"cycle error (non-fatal): {e}")
        time.sleep(max(30.0, a.interval_min * 60))


if __name__ == "__main__":
    main()
