"""
The full 2x2 comparison: Kronos-small vs. Kronos-base, each with and
without the indicator filter + hysteresis, on the same two test windows
used in pnl_backtest_filtered.py (a recent uptrend month and the June
2026 down-trend month).

This reuses the exact indicator/hysteresis/simulate logic from
pnl_backtest_filtered.py -- nothing about the strategy itself is
redefined here, this script just runs it once per model and puts all
four resulting equity curves (Small-Unfiltered, Small-Filtered,
Base-Unfiltered, Base-Filtered) on the same chart against Buy & Hold, so
they're directly comparable side by side instead of spread across
separate charts/scripts.

Charts are drawn at 0.20% slippage only (the lower, more realistic of
the two scenarios) to keep each chart readable -- the summary CSV/table
still includes both 0.20% and 0.54% for every combination.

Run after both backtest_btc_accuracy.py runs (small + base) have
produced their *_results.csv files:
    .venv/bin/python examples/pnl_compare_all.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from pnl_backtest_filtered import (
    add_indicators, resolve_window, simulate,
    INITIAL_CAPITAL, SLIPPAGE_SCENARIOS, MIN_HOLD_BARS, WINDOWS,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")
CHART_SLIPPAGE = 0.20  # the scenario actually plotted; both are in the summary table/CSV

MODELS = {
    "Small": OUT_DIR,
    "Base": os.path.join(OUT_DIR, "kronos-base"),
}

INTERVALS = {
    "1d": "Daily",
    "1h": "Hourly",
    "15m": "15-Minute",
}

LINE_STYLES = {
    ("Small", False): dict(color="#1f77b4", linestyle="-", linewidth=1.4, label="Small, Unfiltered"),
    ("Small", True): dict(color="#1f77b4", linestyle="--", linewidth=1.8, label="Small, Filtered+Hysteresis"),
    ("Base", False): dict(color="#d62728", linestyle="-", linewidth=1.4, label="Base, Unfiltered"),
    ("Base", True): dict(color="#d62728", linestyle="--", linewidth=1.8, label="Base, Filtered+Hysteresis"),
}


def plot_all(interval, label, window_label, window_slug, curves, bh_df):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for (model_name, use_filter), df in curves.items():
        style = LINE_STYLES[(model_name, use_filter)]
        ax.plot(df["timestamp"], df["equity"], **style)
    ax.plot(bh_df["timestamp"], bh_df["buy_hold_equity"], label="Buy & Hold (no trading)",
            color="gray", linestyle=":", linewidth=1.6)
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"BTC/USDT {label} — Small vs. Base, Filtered vs. Unfiltered "
                 f"({window_label}, {CHART_SLIPPAGE:.2f}% slippage, start ${INITIAL_CAPITAL:,.0f})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Account Value (USDT)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"btc_{interval}_pnl_compare_all_{window_slug}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    all_rows = []
    for window_label, start, end in WINDOWS:
        window_slug = window_label.lower().split(" ")[0].replace("(", "").replace(")", "")
        print("\n" + "#" * 100)
        print(f"# WINDOW: {window_label}")
        print("#" * 100)

        for interval, label in INTERVALS.items():
            print(f"\n=== {label} ({interval}) — {window_label} ===")
            curves = {}
            bh_df = None
            any_data = False

            for model_name, model_dir in MODELS.items():
                csv_path = os.path.join(model_dir, f"btc_{interval}_results.csv")
                if not os.path.exists(csv_path):
                    print(f"  [skip] {model_name}: {csv_path} not found")
                    continue
                full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                full_df = add_indicators(full_df)
                window_df = resolve_window(full_df, start, end)
                window_df = window_df.dropna(subset=["sma20", "macd_hist"]).reset_index(drop=True)
                if len(window_df) < 5:
                    print(f"  [skip] {model_name}: only {len(window_df)} bars in window")
                    continue
                any_data = True

                for use_filter in (False, True):
                    strategy_name = "Filtered+Hysteresis" if use_filter else "Unfiltered"
                    for slip in SLIPPAGE_SCENARIOS:
                        sim_df, metrics = simulate(window_df, slip, use_filter=use_filter)
                        if slip == CHART_SLIPPAGE:
                            curves[(model_name, use_filter)] = sim_df
                            if bh_df is None:
                                bh_df = sim_df
                        print(f"  {model_name:<6} {strategy_name:<20} slip={slip:.2f}%: "
                              f"final=${metrics['final_equity']:>9,.2f} ({metrics['total_return_pct']:+7.2f}%), "
                              f"trades={metrics['num_trades']:>4}, win_rate={metrics['win_rate_pct']:5.1f}%")
                        all_rows.append({
                            "Window": window_label, "Interval": label, "Model": model_name,
                            "Strategy": strategy_name, "Slippage %": slip, **metrics,
                        })

            if any_data and bh_df is not None:
                print(f"  Buy & Hold: final=${bh_df['buy_hold_equity'].iloc[-1]:,.2f} "
                      f"({(bh_df['buy_hold_equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100:+.2f}%)")
                path = plot_all(interval, label, window_label, window_slug, curves, bh_df)
                print(f"  Saved chart: {path}")

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUT_DIR, "pnl_compare_all_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 110)
    print(f"Full Summary — starting capital ${INITIAL_CAPITAL:,.0f} each, MIN_HOLD_BARS={MIN_HOLD_BARS}")
    print("=" * 110)
    display_cols = ["Window", "Interval", "Model", "Strategy", "Slippage %",
                     "final_equity", "total_return_pct", "num_trades", "win_rate_pct", "buy_hold_return_pct"]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
