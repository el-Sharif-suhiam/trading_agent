import argparse, os, time
import ccxt
import pandas as pd


def fetch_symbol(exchange, symbol, timeframe, since_ms=None, limit=1000):
    rows = []
    cursor = since_ms
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        nxt = last + tf_ms
        if cursor is not None and nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    tf = pd.Timedelta(milliseconds=tf_ms)
    df = df[df.datetime + tf <= now]
    return df[["datetime","open","high","low","close","volume"]].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--symbols", nargs="+", required=True)
    d.add_argument("--timeframe", default="1h")
    d.add_argument("--since", default="2020-01-01")
    d.add_argument("--raw-dir", default="data/raw")
    d.add_argument("--skip-until", default="",
                   help="symbol=YYYY-MM-DD pairs; drop candles before this date for that symbol "
                        "(e.g. ICP/USDT=2022-01-11 to skip its first 8 listed months)")
    a = p.parse_args()
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    ex.load_markets()
    os.makedirs(a.raw_dir, exist_ok=True)
    since = int(pd.Timestamp(a.since, tz="UTC").timestamp() * 1000)
    skips = {}
    for item in (a.skip_until or []):
        sym, date = item.split("=")
        skips[sym.upper()] = pd.Timestamp(date, tz="UTC")
    for symbol in a.symbols:
        if symbol not in ex.markets or not ex.markets[symbol].get("spot", True):
            raise ValueError(f"Unsupported Binance Spot symbol: {symbol}")
        df = fetch_symbol(ex, symbol, a.timeframe, since)
        if symbol.upper() in skips:
            before = len(df)
            df = df[df.datetime >= skips[symbol.upper()]].reset_index(drop=True)
            print(f"{symbol}: dropped {before - len(df)} candles before {skips[symbol.upper()].date()}")
        name = symbol.replace("/", "_") + ".csv"
        df.to_csv(os.path.join(a.raw_dir, name), index=False)
        print(symbol, len(df), df.datetime.iloc[0], df.datetime.iloc[-1])

if __name__ == "__main__": main()
