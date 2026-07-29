"""
Fetch hourly OHLCV history for 5 coins from Binance, 2018-11-01 through today.

Hourly counterpart to fetch_multi_coin_data.py (which fetches daily). The
walk-forward hourly config needs context_len=168 (7 days) of history before
its first forecast, and we want the first SCORED bar on/after 2019-01-01, so
2018-11-01 gives ~2 months of buffer -- far more than the 7 days required.

Run with:
    .venv/bin/python examples/rl_trading/fetch_multi_coin_data_1h.py
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
INTERVAL = "1h"
START_DATE = "2018-11-01"
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "amount", "trades", "taker_base", "taker_quote", "ignore",
]


def fetch_klines(symbol, interval, start_ms, end_ms):
    url = "https://api.binance.com/api/v3/klines"
    out = []
    cur = start_ms
    while cur < end_ms:
        params = dict(symbol=symbol, interval=interval, startTime=cur, endTime=end_ms, limit=1000)
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cur = batch[-1][0] + 1
        time.sleep(0.2)
    return out


def klines_to_df(raw):
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = df[c].astype(float)
    df = (
        df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        .drop_duplicates("timestamps")
        .sort_values("timestamps")
        .reset_index(drop=True)
    )
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    summary = []
    for symbol in SYMBOLS:
        print(f"Fetching {symbol} hourly klines from {START_DATE} to now ...")
        raw = fetch_klines(symbol, INTERVAL, start_ms, end_ms)
        df = klines_to_df(raw)
        out_path = os.path.join(OUT_DIR, f"{symbol}_1h_raw.csv")
        df.to_csv(out_path, index=False)

        first_ts = df["timestamps"].iloc[0] if len(df) else None
        last_ts = df["timestamps"].iloc[-1] if len(df) else None
        buffer_hours = None
        if first_ts is not None:
            buffer_hours = (pd.Timestamp("2019-01-01") - first_ts).total_seconds() / 3600
        print(f"  {symbol}: {len(df)} rows, {first_ts} -> {last_ts}, "
              f"buffer before 2019-01-01: {buffer_hours:.0f} hours")
        summary.append({
            "symbol": symbol, "rows": len(df), "first": first_ts, "last": last_ts,
            "buffer_hours_before_2019": buffer_hours,
        })

    print("\n" + "=" * 70)
    print("Fetch summary")
    print("=" * 70)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    short = summary_df[summary_df["buffer_hours_before_2019"] < 168]
    if len(short):
        print("\nWARNING: these symbols have < 168 hours of buffer before "
              "2019-01-01, context warm-up may be short:")
        print(short.to_string(index=False))
    else:
        print("\nAll symbols have >= 168 hours of buffer before 2019-01-01.")


if __name__ == "__main__":
    main()
