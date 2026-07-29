"""
Indicator-filtered variant of pnl_backtest.py.

Background: the plain directional strategy in pnl_backtest.py trades every
single bar based on Kronos's predicted direction, which our accuracy
backtest showed is only ~49-51% accurate (a coin flip) -- so trading every
bar just bleeds the account to slippage. This script tests the cheapest
fix from that finding: don't change Kronos at all, just add a filter on
top of its forecasts using two independent, classic technical indicators
computed from the REAL price series (not the model's forecast):

  - Trend filter: actual close vs. its 20-period SMA (Simple Moving
    Average). Price above the SMA => uptrend => only longs allowed.
    Price below => downtrend => only shorts allowed.
  - Momentum filter: MACD(12,26,9) histogram sign. MACD (Moving Average
    Convergence/Divergence) is fast-EMA minus slow-EMA; the histogram is
    MACD minus its own signal line. Positive histogram => bullish
    momentum, negative => bearish.

A trade is only taken when Kronos's directional call AGREES with both
filters, and the predicted move is at least MIN_MOVE_PCT (skips
near-zero, noise-level predictions). Otherwise the bar is flat (no
position, no slippage cost).

HYSTERESIS: an earlier version of this filter without hysteresis was
found to actually INCREASE trade count at 15-minute granularity (131 ->
393 trades) because the position kept flipping in and out of flat as the
agree/disagree check flickered bar to bar -- the extra slippage from that
churn wiped out the filter's win-rate gains. To fix this, once a
non-flat position is opened it must be held for at least MIN_HOLD_BARS
bars before the filter is allowed to change it again (exit to flat or
flip direction) -- a simple debounce/cooldown, applied as a small state
machine in `apply_hysteresis`.

VOLUME FILTERS (added on top of the original trend+MACD combo above --
that combo is untouched and still runs exactly as before as the
"Filtered+Hysteresis" curve; these add a THIRD curve,
"Filtered+Volume+Hysteresis", gated by the USE_CMF_FILTER /
USE_VOLUME_VETO toggles below):
  - Chaikin Money Flow (CMF, 20-period): a bounded [-1, 1] buying/selling
    pressure indicator computed from high/low/close/volume. Its sign is
    already directional (>0 buying pressure, <0 selling), so it slots in
    as a FOURTH required sign-agreement gate alongside Kronos/trend/MACD,
    same shape as the existing filters.
  - Volume regime (rising/falling/flat): current bar's volume vs. its own
    20-period rolling mean, classified rising (+1)/falling(-1)/flat (0)
    using a +/-15% band. Unlike CMF this has no natural "direction" to
    agree with Kronos on -- flat volume just means low conviction behind
    the move either way -- so it's used as a VETO instead: force the
    position flat whenever volume is in the flat band, regardless of what
    every other filter says.
Volume/high/low aren't in the *_results.csv (that file only has
timestamp/actual/predicted/block), so they're re-fetched fresh from the
same Binance klines endpoint backtest_btc_accuracy.py uses and merged in
by timestamp -- see `fetch_volume_data`.

Indicators are computed over each interval's FULL available history (so
SMA/MACD have a proper warm-up), then everything is sliced down to a
specific test window for the actual comparison -- run as a fresh $10,000
backtest starting at the beginning of that window, so it's a clean read
on "how would this have done over that period" rather than a
continuation of a longer equity curve. Two windows are tested:
  - "Recent" -- the most recent TEST_WINDOW_DAYS days (an uptrend in this
    data, BTC roughly +5% over the window).
  - "Down-Trend" -- June 2026, a clear down-trending month in this same
    BTC/USDT history (buy & hold roughly -20% over the window), so the
    filter gets tested in a regime opposite to the one it was designed
    against.

Run after backtest_btc_accuracy.py has produced the *_results.csv files:
    .venv/bin/python examples/pnl_backtest_filtered.py
"""
import os
import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_SCENARIOS = [0.20, 0.54]  # percent per side, same as pnl_backtest.py
TEST_WINDOW_DAYS = 30
MIN_MOVE_PCT = 0.05  # skip trades where Kronos's predicted move is smaller than this (%)
MIN_HOLD_BARS = 3  # once in a position, hold at least this many bars before the filter can change it

# --- Volume filter toggles (additive -- both default on, but flip either to
# False to fall back toward the original trend+MACD-only behavior without
# touching any logic below) ---
USE_CMF_FILTER = True     # require CMF sign to also agree (4th sign-agreement gate)
USE_VOLUME_VETO = True    # force flat whenever volume is in its "flat" band
CMF_WINDOW = 20
VOL_MA_WINDOW = 20
VOL_FLAT_BAND = 0.15  # +/-15% around volume's own rolling mean counts as "flat"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

INTERVALS = {
    "1d": "Daily",
    "1h": "Hourly",
    "15m": "15-Minute",
}

# (start, end) are inclusive-ish bounds applied to the timestamp column; end=None means "through the latest data"
WINDOWS = [
    ("Recent", None, None),  # resolved to (latest - TEST_WINDOW_DAYS, latest] per-interval at runtime
    ("Down-Trend (Jun 2026)", "2026-06-01", "2026-07-01"),
]


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def macd_histogram(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def fetch_volume_data(interval, start_ts, end_ts):
    """Pull high/low/volume from Binance klines for CMF + volume-regime --
    not present in *_results.csv, so fetched fresh here (same endpoint/symbol
    as backtest_btc_accuracy.py) and merged onto the results by timestamp."""
    start_ms = int(pd.Timestamp(start_ts).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_ts).timestamp() * 1000) + 1
    out = []
    cur = start_ms
    while cur < end_ms:
        params = dict(symbol="BTCUSDT", interval=interval, startTime=cur, endTime=end_ms, limit=1000)
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cur = batch[-1][0] + 1
        time.sleep(0.2)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "amount", "trades", "taker_base", "taker_quote", "ignore"]
    vol_df = pd.DataFrame(out, columns=cols)
    vol_df["timestamp"] = pd.to_datetime(vol_df["open_time"], unit="ms")
    for c in ["high", "low", "volume"]:
        vol_df[c] = vol_df[c].astype(float)
    return (
        vol_df[["timestamp", "high", "low", "volume"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def cmf(df, window=20):
    """Chaikin Money Flow: bounded [-1, 1] buying/selling pressure. Sign is
    directional (>0 buying pressure, <0 selling) -- same shape as macd_dir."""
    high, low, close, volume = df["high"], df["low"], df["actual"], df["volume"]
    span = (high - low).replace(0, np.nan)
    money_flow_mult = (((close - low) - (high - close)) / span).fillna(0.0)
    money_flow_volume = money_flow_mult * volume
    return money_flow_volume.rolling(window).sum() / volume.rolling(window).sum()


def volume_regime(df, window=20, flat_band=0.15):
    """Classify each bar's volume vs. its own rolling mean: +1 rising,
    -1 falling, 0 flat (within +/-flat_band). NaN during warm-up."""
    vol_ma = df["volume"].rolling(window).mean()
    ratio = df["volume"] / vol_ma
    regime = pd.Series(np.nan, index=df.index)
    regime[ratio > 1 + flat_band] = 1
    regime[ratio < 1 - flat_band] = -1
    regime[(ratio >= 1 - flat_band) & (ratio <= 1 + flat_band)] = 0
    return regime


def add_indicators(df):
    df = df.copy()
    df["sma20"] = sma(df["actual"], 20)
    df["macd_hist"] = macd_histogram(df["actual"])
    # --- volume filters, additive on top of the two above ---
    df["cmf"] = cmf(df, CMF_WINDOW)
    df["vol_regime"] = volume_regime(df, VOL_MA_WINDOW, VOL_FLAT_BAND)
    return df


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


def simulate(df, slippage_pct, use_filter, use_cmf=False, use_volume_veto=False):
    df = df.copy()
    df["prev_actual"] = df["actual"].shift(1)
    df = df.dropna(subset=["prev_actual"]).reset_index(drop=True)

    pred_ret = (df["predicted"] - df["prev_actual"]) / df["prev_actual"]
    kronos_dir = np.sign(pred_ret)

    if use_filter:
        trend_dir = np.sign(df["actual"] - df["sma20"])
        macd_dir = np.sign(df["macd_hist"])
        agree = (kronos_dir == trend_dir) & (kronos_dir == macd_dir)
        if use_cmf:
            # 4th sign-agreement gate, same shape as trend_dir/macd_dir above
            cmf_dir = np.sign(df["cmf"])
            agree = agree & (kronos_dir == cmf_dir)
        big_enough = pred_ret.abs() >= (MIN_MOVE_PCT / 100.0)
        desired_position = np.where(agree & big_enough, kronos_dir, 0)
        if use_volume_veto:
            # low-conviction veto -- flat volume forces flat position
            # regardless of what the direction-agreement filters say
            desired_position = np.where(df["vol_regime"].values == 0, 0, desired_position)
        df["position"] = apply_hysteresis(desired_position, MIN_HOLD_BARS)
    else:
        df["position"] = kronos_dir

    df["bar_return"] = df["position"] * (df["actual"] / df["prev_actual"] - 1)

    prev_pos = df["position"].shift(1).fillna(0)
    position_change = (df["position"] - prev_pos).abs()
    df["slippage_cost"] = position_change * (slippage_pct / 100.0)
    df["net_return"] = df["bar_return"] - df["slippage_cost"]

    df["equity"] = INITIAL_CAPITAL * (1 + df["net_return"]).cumprod()
    bh_return = df["actual"] / df["prev_actual"] - 1
    df["buy_hold_equity"] = INITIAL_CAPITAL * (1 + bh_return).cumprod()

    num_trades = int((position_change > 0).sum())
    active_bars = int((df["position"] != 0).sum())
    winning_bars = int(((df["bar_return"] > 0) & (df["position"] != 0)).sum())
    total_slippage_cost = float((df["slippage_cost"] * INITIAL_CAPITAL).sum())

    return df, {
        "final_equity": float(df["equity"].iloc[-1]),
        "total_return_pct": float((df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "buy_hold_final_equity": float(df["buy_hold_equity"].iloc[-1]),
        "buy_hold_return_pct": float((df["buy_hold_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "num_trades": num_trades,
        "active_bars": active_bars,
        "total_bars": len(df),
        "win_rate_pct": 100.0 * winning_bars / active_bars if active_bars else float("nan"),
        "total_slippage_cost_usd": total_slippage_cost,
    }


def plot_equity(interval, label, window_label, window_slug, curves):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, df in curves.items():
        ax.plot(df["timestamp"], df["equity"], label=name, linewidth=1.5)
    any_df = next(iter(curves.values()))
    ax.plot(any_df["timestamp"], any_df["buy_hold_equity"], label="Buy & Hold (no trading)",
            color="gray", linestyle="--", linewidth=1.3)
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"BTC/USDT {label} — Filtered+Hysteresis vs. Unfiltered Kronos Strategy "
                 f"({window_label}, start ${INITIAL_CAPITAL:,.0f})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Account Value (USDT)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"btc_{interval}_pnl_filtered_{window_slug}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def resolve_window(full_df, start, end):
    if start is None and end is None:
        cutoff = full_df["timestamp"].max() - pd.Timedelta(TEST_WINDOW_DAYS, unit="D")
        return full_df[full_df["timestamp"] > cutoff].reset_index(drop=True)
    mask = (full_df["timestamp"] >= pd.Timestamp(start)) & (full_df["timestamp"] < pd.Timestamp(end))
    return full_df[mask].reset_index(drop=True)


def main():
    all_rows = []
    volume_cache = {}  # interval -> volume_df, avoids re-fetching per window

    vol_label_parts = []
    if USE_CMF_FILTER:
        vol_label_parts.append("CMF")
    if USE_VOLUME_VETO:
        vol_label_parts.append("VolVeto")
    vol_variant_label = (
        f"Filtered+{'+'.join(vol_label_parts)}+Hysteresis" if vol_label_parts
        else "Filtered+Hysteresis (volume off)"
    )

    for window_label, start, end in WINDOWS:
        window_slug = window_label.lower().split(" ")[0].replace("(", "").replace(")", "")
        print("\n" + "#" * 90)
        print(f"# WINDOW: {window_label}")
        print("#" * 90)

        for interval, label in INTERVALS.items():
            csv_path = os.path.join(OUT_DIR, f"btc_{interval}_results.csv")
            full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])

            if interval not in volume_cache:
                print(f"  Fetching volume data for {label} ({interval}) ...")
                volume_cache[interval] = fetch_volume_data(
                    interval, full_df["timestamp"].min(), full_df["timestamp"].max()
                )
            full_df = full_df.merge(volume_cache[interval], on="timestamp", how="left")
            full_df = add_indicators(full_df)

            window_df = resolve_window(full_df, start, end)
            required_cols = ["sma20", "macd_hist"]
            if USE_CMF_FILTER:
                required_cols.append("cmf")
            if USE_VOLUME_VETO:
                required_cols.append("vol_regime")
            window_df = window_df.dropna(subset=required_cols).reset_index(drop=True)
            if len(window_df) < 5:
                print(f"\n=== {label} ({interval}) — {window_label}: skipped, only {len(window_df)} bars ===")
                continue

            print(f"\n=== {label} ({interval}) — {window_label}, {len(window_df)} bars ===")
            curves = {}
            for slip in SLIPPAGE_SCENARIOS:
                base_df, base_m = simulate(window_df, slip, use_filter=False)
                filt_df, filt_m = simulate(window_df, slip, use_filter=True)
                vol_df_sim, vol_m = simulate(
                    window_df, slip, use_filter=True,
                    use_cmf=USE_CMF_FILTER, use_volume_veto=USE_VOLUME_VETO,
                )
                curves[f"Unfiltered ({slip:.2f}% slip)"] = base_df
                curves[f"Filtered+Hysteresis ({slip:.2f}% slip)"] = filt_df
                curves[f"{vol_variant_label} ({slip:.2f}% slip)"] = vol_df_sim

                print(f"  Slippage {slip:.2f}%:")
                print(f"    Unfiltered:         final=${base_m['final_equity']:,.2f} ({base_m['total_return_pct']:+.2f}%), "
                      f"trades={base_m['num_trades']}, active_bars={base_m['active_bars']}/{base_m['total_bars']}, "
                      f"win_rate={base_m['win_rate_pct']:.1f}%")
                print(f"    Filtered+Hysteresis: final=${filt_m['final_equity']:,.2f} ({filt_m['total_return_pct']:+.2f}%), "
                      f"trades={filt_m['num_trades']}, active_bars={filt_m['active_bars']}/{filt_m['total_bars']}, "
                      f"win_rate={filt_m['win_rate_pct']:.1f}%")
                print(f"    {vol_variant_label}: final=${vol_m['final_equity']:,.2f} ({vol_m['total_return_pct']:+.2f}%), "
                      f"trades={vol_m['num_trades']}, active_bars={vol_m['active_bars']}/{vol_m['total_bars']}, "
                      f"win_rate={vol_m['win_rate_pct']:.1f}%")

                all_rows.append({"Window": window_label, "Interval": label, "Slippage %": slip,
                                  "Strategy": "Unfiltered", **base_m})
                all_rows.append({"Window": window_label, "Interval": label, "Slippage %": slip,
                                  "Strategy": "Filtered+Hysteresis", **filt_m})
                all_rows.append({"Window": window_label, "Interval": label, "Slippage %": slip,
                                  "Strategy": vol_variant_label, **vol_m})

            print(f"  Buy & Hold: final=${base_m['buy_hold_final_equity']:,.2f} ({base_m['buy_hold_return_pct']:+.2f}%)")
            path = plot_equity(interval, label, window_label, window_slug, curves)
            print(f"  Saved chart: {path}")

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUT_DIR, "pnl_filtered_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print(f"Summary — starting capital ${INITIAL_CAPITAL:,.0f} each, MIN_HOLD_BARS={MIN_HOLD_BARS}")
    print("=" * 100)
    display_cols = ["Window", "Interval", "Slippage %", "Strategy", "final_equity", "total_return_pct",
                     "num_trades", "active_bars", "total_bars", "win_rate_pct", "buy_hold_return_pct"]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
