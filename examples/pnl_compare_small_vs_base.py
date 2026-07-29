"""
Run the same naive directional P&L strategy from pnl_backtest.py on both
Kronos-small and Kronos-base's predictions (full 6-month backtest, same
walk-forward setup) and compare them side by side.

Strategy (unchanged from pnl_backtest.py): at each bar, go long if the
model predicted a higher close than the last known real close, short if
lower. Apply that bar's real return, minus a slippage/fee cost whenever
the position changes.

Reads:
  Kronos-small: examples/accuracy_backtest_output/btc_{interval}_results.csv
  Kronos-base:  examples/accuracy_backtest_output/kronos-base/btc_{interval}_results.csv

Run after both backtest_btc_accuracy.py runs (small + base, see
KRONOS_MODEL_NAME/KRONOS_OUT_SUBDIR env vars in that script) have
produced their *_results.csv files:
    .venv/bin/python examples/pnl_compare_small_vs_base.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from pnl_backtest import simulate, INITIAL_CAPITAL, SLIPPAGE_SCENARIOS  # reuse the exact same strategy logic

OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")

MODELS = {
    "Kronos-small": OUT_DIR,
    "Kronos-base": os.path.join(OUT_DIR, "kronos-base"),
}

INTERVALS = {
    "1d": "Daily",
    "1h": "Hourly",
    "15m": "15-Minute",
}


def plot_comparison(interval, label, curves, bh_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, df in curves.items():
        ax.plot(df["timestamp"], df["equity"], label=name, linewidth=1.3)
    ax.plot(bh_df["timestamp"], bh_df["buy_hold_equity"], label="Buy & Hold (no trading)",
            color="gray", linestyle="--", linewidth=1.3)
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"BTC/USDT {label} — Kronos-small vs. Kronos-base Directional Strategy P&L "
                 f"(6 months, start ${INITIAL_CAPITAL:,.0f})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Account Value (USDT)")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"btc_{interval}_pnl_small_vs_base.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    all_rows = []
    for interval, label in INTERVALS.items():
        print(f"\n=== {label} ({interval}) ===")
        curves = {}
        bh_df = None
        for model_name, model_dir in MODELS.items():
            csv_path = os.path.join(model_dir, f"btc_{interval}_results.csv")
            if not os.path.exists(csv_path):
                print(f"  [skip] {model_name}: {csv_path} not found")
                continue
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])

            for slip in SLIPPAGE_SCENARIOS:
                sim_df, metrics = simulate(df, slip)
                curves[f"{model_name} ({slip:.2f}% slip)"] = sim_df
                if bh_df is None:
                    bh_df = sim_df
                print(f"  {model_name:<13} slippage {slip:.2f}%: final=${metrics['final_equity']:,.2f} "
                      f"({metrics['total_return_pct']:+.2f}%), trades={metrics['num_trades']}, "
                      f"win_rate={metrics['win_rate_pct']:.1f}%")
                all_rows.append({"Interval": label, "Model": model_name, "Slippage %": slip, **metrics})

        if bh_df is not None:
            bh_metrics_row = all_rows[-1]
            print(f"  {'Buy & Hold':<13}                : final=${bh_metrics_row['buy_hold_final_equity']:,.2f} "
                  f"({bh_metrics_row['buy_hold_return_pct']:+.2f}%)")
            path = plot_comparison(interval, label, curves, bh_df)
            print(f"  Saved chart: {path}")

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUT_DIR, "pnl_small_vs_base_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print(f"Summary — starting capital ${INITIAL_CAPITAL:,.0f} each, full 6-month backtest")
    print("=" * 100)
    display_cols = ["Interval", "Model", "Slippage %", "final_equity", "total_return_pct",
                     "num_trades", "win_rate_pct", "buy_hold_return_pct"]
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
