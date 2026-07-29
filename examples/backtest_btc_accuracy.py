"""
Walk-forward accuracy backtest for Kronos on BTC/USDT.

For each of three granularities (daily, hourly, 15-minute), this script:
  1. Pulls real OHLCV data from Binance's public REST API covering the last
     TEST_DAYS days plus enough prior history to serve as model context.
  2. Slides a lookback window forward in non-overlapping blocks. For each
     block, Kronos is given the preceding `context_len` bars and asked to
     forecast the next `block_len` bars (a real walk-forward test — the
     model never sees the bars it's being scored on).
  3. Stitches the per-block predictions back into one continuous series,
     compares it against the actual closing prices, and saves an
     actual-vs-predicted chart (plus an error-% subplot) per granularity.

Run with:
    .venv/bin/python examples/backtest_btc_accuracy.py
"""
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

SYMBOL = "BTCUSDT"
TEST_DAYS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")

# context_len: bars of history fed to the model before each forecast block
# block_len:   bars forecast per block (and the stride between blocks)
INTERVAL_CONFIG = {
    "1d": dict(label="Daily", bars_per_day=1, context_len=90, block_len=5, ms_per_bar=86_400_000),
    "1h": dict(label="Hourly", bars_per_day=24, context_len=168, block_len=24, ms_per_bar=3_600_000),
    "15m": dict(label="15-Minute", bars_per_day=96, context_len=192, block_len=96, ms_per_bar=900_000),
}

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


def build_blocks(df, context_len, block_len, num_blocks):
    df_list, xt_list, yt_list, actual_list = [], [], [], []
    for i in range(num_blocks):
        s = i * block_len
        x_df = df.loc[s:s + context_len - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.loc[s:s + context_len - 1, "timestamps"]
        y_ts = df.loc[s + context_len:s + context_len + block_len - 1, "timestamps"]
        y_actual = df.loc[s + context_len:s + context_len + block_len - 1, "close"].values
        df_list.append(x_df)
        xt_list.append(x_ts)
        yt_list.append(y_ts)
        actual_list.append(y_actual)
    return df_list, xt_list, yt_list, actual_list


def run_interval_backtest(predictor, interval, cfg):
    test_bars = TEST_DAYS * cfg["bars_per_day"]
    context_len, block_len = cfg["context_len"], cfg["block_len"]
    num_blocks = test_bars // block_len
    total_needed = context_len + num_blocks * block_len

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(total_needed * cfg["ms_per_bar"] * 1.3)

    raw = fetch_klines(SYMBOL, interval, start_ms, end_ms)
    df = klines_to_df(raw)
    df = df.tail(total_needed).reset_index(drop=True)
    if len(df) < total_needed:
        raise RuntimeError(f"[{interval}] only got {len(df)} bars, needed {total_needed}")

    df_list, xt_list, yt_list, actual_list = build_blocks(df, context_len, block_len, num_blocks)

    print(f"  [{interval}] forecasting {num_blocks} blocks x {block_len} bars "
          f"(context={context_len}) ...")
    pred_dfs = predictor.predict_batch(
        df_list, xt_list, yt_list, pred_len=block_len,
        T=1.0, top_p=0.9, sample_count=1, verbose=False,
    )

    ts, actual, predicted, block_id = [], [], [], []
    for b, (y_ts, y_actual, pred_df) in enumerate(zip(yt_list, actual_list, pred_dfs)):
        ts.extend(list(y_ts))
        actual.extend(list(y_actual))
        predicted.extend(list(pred_df["close"].values))
        block_id.extend([b] * len(y_ts))

    result = pd.DataFrame(
        {"timestamp": ts, "actual": actual, "predicted": predicted, "block": block_id}
    ).sort_values("timestamp").reset_index(drop=True)
    return result


def compute_metrics(result):
    err = result["predicted"] - result["actual"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / result["actual"])) * 100)
    corr = float(np.corrcoef(result["actual"], result["predicted"])[0, 1])

    # Directional accuracy computed within each block only (first bar of a
    # block has no prior predicted bar to diff against within that block).
    correct, total = 0, 0
    for _, g in result.groupby("block"):
        if len(g) < 2:
            continue
        actual_dir = np.sign(g["actual"].diff().dropna())
        pred_dir = np.sign(g["predicted"].diff().dropna())
        correct += int((actual_dir.values == pred_dir.values).sum())
        total += len(actual_dir)
    dir_acc = 100.0 * correct / total if total else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "DirectionalAccuracy": dir_acc, "Correlation": corr}


def plot_result(result, metrics, interval, cfg):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(result["timestamp"], result["actual"], label="Actual", color="#1f77b4", linewidth=1.5)
    ax1.plot(result["timestamp"], result["predicted"], label="Kronos Prediction",
              color="#d62728", linewidth=1.2, alpha=0.85)
    ax1.set_title(f"BTC/USDT {cfg['label']} — Kronos Walk-Forward Backtest (last {TEST_DAYS} days)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Close Price (USDT)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    text = (
        f"MAE: {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"MAPE: {metrics['MAPE']:.2f}%\n"
        f"Directional Acc: {metrics['DirectionalAccuracy']:.1f}%\n"
        f"Correlation: {metrics['Correlation']:.3f}"
    )
    ax1.text(0.01, 0.98, text, transform=ax1.transAxes, va="top", fontsize=10,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    pct_err = (result["predicted"] - result["actual"]) / result["actual"] * 100
    ax2.plot(result["timestamp"], pct_err, color="#555555", linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("% Error")
    ax2.set_xlabel("Time")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"btc_{interval}_backtest.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    print("Loading Kronos-base + Kronos-Tokenizer-base ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    summary = {}
    for interval, cfg in INTERVAL_CONFIG.items():
        print(f"\n=== {cfg['label']} backtest ({interval}) ===")
        result = run_interval_backtest(predictor, interval, cfg)
        metrics = compute_metrics(result)
        result.to_csv(os.path.join(OUT_DIR, f"btc_{interval}_results.csv"), index=False)
        path = plot_result(result, metrics, interval, cfg)
        summary[cfg["label"]] = metrics
        print(f"  Saved chart: {path}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(pd.DataFrame(summary).T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nAll outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
