"""
Multi-coin daily walk-forward Kronos-small backtest (2019-2026) that produces
a per-trade log for 5 coins, for later use as meta-labeling training data.

For each symbol's raw CSV (produced by fetch_multi_coin_data.py, starting
2018-06-01 so there's >90 days of context buffer before scoring begins):
  1. Build non-overlapping walk-forward blocks (context_len=90, block_len=5)
     starting far enough back that the FIRST scored bar lands on/after
     2019-01-01 -- the 2018-06 to 2019-01 buffer is only ever used as model
     context, never scored/traded on.
  2. Forecast each block with Kronos-small (same predict_batch call shape as
     ../backtest_btc_accuracy.py, read-only reference, not imported), stitch
     into one continuous (timestamp, actual, predicted) series per symbol.
  3. Compute SMA20 + MACD(12,26,9) histogram from the symbol's full actual
     price history (so indicators are properly warmed up before 2019-01-01),
     then apply the Filtered+Hysteresis strategy: only take Kronos's
     directional call when it agrees with both the SMA20 trend sign and the
     MACD histogram sign, and the predicted move is >= 0.05%; otherwise stay
     flat. Once non-flat, hold >= 3 bars before the filter can change the
     position again (see apply_hysteresis -- reimplemented here from the
     algorithm in ../pnl_backtest_filtered.py, read-only reference, not
     imported).
  4. Walk the resulting per-bar position series and extract one row per
     contiguous non-zero-position segment (a trade), with the entry
     conditions (predicted move, SMA distance, MACD state) and the realized
     return net of 0.20%-per-side slippage.
  5. Concatenate all 5 symbols' trade logs into one combined file.

This is a long-running CPU-only research script (5 coins x ~7 years of
daily bars x Kronos-small inference, no GPU) -- expect it to take a while;
progress is printed per symbol and per prediction-batch chunk.

Run with:
    .venv/bin/python examples/rl_trading/run_walkforward.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

CONTEXT_LEN = 90
BLOCK_LEN = 5
SCORE_START = "2019-01-01"

SLIPPAGE_PCT = 0.20      # percent per side
MIN_MOVE_PCT = 0.05      # skip trades where Kronos's predicted move is smaller than this (%)
MIN_HOLD_BARS = 3        # once in a position, hold at least this many bars before the filter can change it

PREDICT_BATCH_SIZE = 16  # blocks per predict_batch() call -- bounds memory/time per call on CPU


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def macd_histogram(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def apply_hysteresis(desired_position, min_hold_bars):
    """Debounce a desired-position series: once non-flat, hold for at
    least min_hold_bars bars before the filter is allowed to change it
    again (exit to flat or flip direction). Flat -> non-flat entries are
    always allowed immediately."""
    position = np.zeros(len(desired_position), dtype=int)
    current = 0
    bars_in_position = 0
    for i, desired in enumerate(desired_position):
        if current == 0:
            current = int(desired)
            bars_in_position = 1 if current != 0 else 0
        elif bars_in_position < min_hold_bars:
            bars_in_position += 1
        elif desired != current:
            current = int(desired)
            bars_in_position = 1 if current != 0 else 0
        else:
            bars_in_position += 1
        position[i] = current
    return position


def build_blocks(df, s0, context_len, block_len):
    df_list, xt_list, yt_list, actual_list = [], [], [], []
    s = s0
    n = len(df)
    while s + context_len + block_len <= n:
        x_df = df.loc[s:s + context_len - 1, ["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.loc[s:s + context_len - 1, "timestamps"]
        y_ts = df.loc[s + context_len:s + context_len + block_len - 1, "timestamps"]
        y_actual = df.loc[s + context_len:s + context_len + block_len - 1, "close"].values
        df_list.append(x_df)
        xt_list.append(x_ts)
        yt_list.append(y_ts)
        actual_list.append(y_actual)
        s += block_len
    return df_list, xt_list, yt_list, actual_list


def run_symbol_forecast(predictor, symbol):
    raw_path = os.path.join(OUT_DIR, f"{symbol}_1d_raw.csv")
    raw_df = pd.read_csv(raw_path, parse_dates=["timestamps"]).reset_index(drop=True)

    on_or_after = raw_df.index[raw_df["timestamps"] >= pd.Timestamp(SCORE_START)]
    if len(on_or_after) == 0:
        raise RuntimeError(f"{symbol}: no bars on/after {SCORE_START}")
    idx0 = on_or_after[0]
    s0 = idx0 - CONTEXT_LEN
    if s0 < 0:
        raise RuntimeError(
            f"{symbol}: not enough history before {SCORE_START} "
            f"(need {CONTEXT_LEN} bars of context, only have {idx0})"
        )

    df_list, xt_list, yt_list, actual_list = build_blocks(raw_df, s0, CONTEXT_LEN, BLOCK_LEN)
    num_blocks = len(df_list)
    print(f"  [{symbol}] {num_blocks} blocks x {BLOCK_LEN} bars (context={CONTEXT_LEN})")

    ts, actual, predicted = [], [], []
    t0 = time.time()
    for start in range(0, num_blocks, PREDICT_BATCH_SIZE):
        chunk_df = df_list[start:start + PREDICT_BATCH_SIZE]
        chunk_xt = xt_list[start:start + PREDICT_BATCH_SIZE]
        chunk_yt = yt_list[start:start + PREDICT_BATCH_SIZE]
        chunk_actual = actual_list[start:start + PREDICT_BATCH_SIZE]
        pred_dfs = predictor.predict_batch(
            chunk_df, chunk_xt, chunk_yt, pred_len=BLOCK_LEN,
            T=1.0, top_p=0.9, sample_count=1, verbose=False,
        )
        for y_ts, y_actual, pred_df in zip(chunk_yt, chunk_actual, pred_dfs):
            ts.extend(list(y_ts))
            actual.extend(list(y_actual))
            predicted.extend(list(pred_df["close"].values))

        done = min(start + PREDICT_BATCH_SIZE, num_blocks)
        elapsed = time.time() - t0
        print(f"    [{symbol}] {done}/{num_blocks} blocks forecast ({elapsed:.0f}s elapsed)")

    result = (
        pd.DataFrame({"timestamp": ts, "actual": actual, "predicted": predicted})
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    print(f"  [{symbol}] scored {len(result)} bars, {result['timestamp'].min()} -> {result['timestamp'].max()}")
    return result, raw_df


def add_indicators_and_strategy(result, raw_df):
    """Compute SMA20/MACD hist from the FULL raw history (proper warm-up
    before 2019-01-01), merge onto the scored series, then apply the
    Filtered+Hysteresis strategy."""
    raw_df = raw_df.copy()
    raw_df["sma20"] = sma(raw_df["close"], 20)
    raw_df["macd_hist"] = macd_histogram(raw_df["close"])
    raw_df["prev_close"] = raw_df["close"].shift(1)

    indicators = raw_df[["timestamps", "sma20", "macd_hist", "prev_close"]].rename(
        columns={"timestamps": "timestamp", "prev_close": "prev_actual"}
    )
    df = result.merge(indicators, on="timestamp", how="left")

    before = len(df)
    df = df.dropna(subset=["sma20", "macd_hist", "prev_actual"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  WARNING: dropped {before - len(df)} rows with missing indicators/prev_actual")

    pred_ret = (df["predicted"] - df["prev_actual"]) / df["prev_actual"]
    kronos_dir = np.sign(pred_ret)
    trend_dir = np.sign(df["actual"] - df["sma20"])
    macd_dir = np.sign(df["macd_hist"])
    agree = (kronos_dir == trend_dir) & (kronos_dir == macd_dir)
    big_enough = pred_ret.abs() >= (MIN_MOVE_PCT / 100.0)
    desired_position = np.where(agree & big_enough, kronos_dir, 0)

    df["position"] = apply_hysteresis(desired_position, MIN_HOLD_BARS)
    df["pred_ret_pct"] = pred_ret * 100
    df["sma_dist_pct"] = (df["actual"] - df["sma20"]) / df["sma20"] * 100
    df["macd_hist_slope"] = df["macd_hist"].diff()
    return df


def extract_trades(df, symbol, slippage_pct):
    """Walk the per-bar position series and turn each contiguous non-zero
    segment into one trade-log row. Entry/exit prices follow the same
    convention as the bar-return formula used elsewhere in this project
    (position_i earns the return from close[i-1] to close[i]): entry_price
    is the close right before the segment starts, exit_price is the close
    of the segment's last held bar. Slippage is charged once on entry and
    once on exit (2 x slippage_pct total)."""
    positions = df["position"].values
    n = len(df)
    trades = []
    i = 0
    while i < n:
        pos = positions[i]
        if pos == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and positions[j + 1] == pos:
            j += 1

        entry_price = df["prev_actual"].iloc[i]
        exit_price = df["actual"].iloc[j]
        direction = int(pos)
        holding_bars = j - i + 1
        price_return = direction * (exit_price / entry_price - 1)
        realized_return_pct = (price_return - 2 * (slippage_pct / 100.0)) * 100

        trades.append({
            "symbol": symbol,
            "entry_timestamp": df["timestamp"].iloc[i],
            "exit_timestamp": df["timestamp"].iloc[j],
            "direction": direction,
            "holding_bars": holding_bars,
            "predicted_move_pct_at_entry": float(df["pred_ret_pct"].iloc[i]),
            "sma_dist_pct_at_entry": float(df["sma_dist_pct"].iloc[i]),
            "macd_hist_at_entry": float(df["macd_hist"].iloc[i]),
            "macd_hist_slope_at_entry": float(df["macd_hist_slope"].iloc[i]),
            "realized_return_pct": float(realized_return_pct),
            "outcome": 1 if realized_return_pct > 0 else 0,
        })
        i = j + 1

    return pd.DataFrame(trades)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading Kronos-small + Kronos-Tokenizer-base ...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    all_trades = []
    summary_rows = []
    overall_t0 = time.time()

    for symbol in SYMBOLS:
        print(f"\n=== {symbol} ===")
        sym_t0 = time.time()

        result, raw_df = run_symbol_forecast(predictor, symbol)
        df = add_indicators_and_strategy(result, raw_df)
        df.to_csv(os.path.join(OUT_DIR, f"{symbol}_1d_results.csv"), index=False)

        trades = extract_trades(df, symbol, SLIPPAGE_PCT)
        trades.to_csv(os.path.join(OUT_DIR, f"{symbol}_1d_trades.csv"), index=False)
        all_trades.append(trades)

        num_trades = len(trades)
        win_rate = 100.0 * trades["outcome"].mean() if num_trades else float("nan")
        total_return = trades["realized_return_pct"].sum() if num_trades else 0.0
        elapsed = time.time() - sym_t0
        print(f"  {symbol}: {num_trades} trades, win_rate={win_rate:.1f}%, "
              f"total_return={total_return:+.2f}% (sum of per-trade returns), "
              f"took {elapsed / 60:.1f} min")
        summary_rows.append({
            "symbol": symbol, "num_trades": num_trades,
            "win_rate_pct": win_rate, "total_return_pct": total_return,
        })

    combined = pd.concat(all_trades, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "all_coins_trade_log.csv"), index=False)

    total_elapsed = time.time() - overall_t0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    combined_trades = len(combined)
    combined_win_rate = 100.0 * combined["outcome"].mean() if combined_trades else float("nan")
    print(f"\nCombined: {combined_trades} trades across {len(SYMBOLS)} coins, "
          f"overall win rate = {combined_win_rate:.1f}%")
    print(f"Total wall-clock time: {total_elapsed / 60:.1f} minutes")
    print(f"Combined trade log: {os.path.join(OUT_DIR, 'all_coins_trade_log.csv')}")


if __name__ == "__main__":
    main()
