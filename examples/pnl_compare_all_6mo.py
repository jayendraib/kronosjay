"""
Same 2x2 comparison as pnl_compare_all.py (Small vs. Base, Filtered vs.
Unfiltered) but run over the FULL 6-month history instead of a 30-day
window -- fills the gap where only the unfiltered strategy had been run
over the full period (pnl_compare_small_vs_base.py) while the filter had
only been tested on two 1-month windows (pnl_compare_all.py).

Run after both backtest_btc_accuracy.py runs (small + base) have
produced their *_results.csv files:
    .venv/bin/python examples/pnl_compare_all_6mo.py
"""
import os
import pandas as pd

from pnl_backtest_filtered import add_indicators, simulate, INITIAL_CAPITAL, SLIPPAGE_SCENARIOS

OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")

MODELS = {
    "Small": OUT_DIR,
    "Base": os.path.join(OUT_DIR, "kronos-base"),
}

INTERVALS = {
    "1d": "Daily",
    "1h": "Hourly",
    "15m": "15-Minute",
}


def main():
    all_rows = []
    for interval, label in INTERVALS.items():
        print(f"\n=== {label} ({interval}) — Full 6 Months ===")
        for model_name, model_dir in MODELS.items():
            csv_path = os.path.join(model_dir, f"btc_{interval}_results.csv")
            if not os.path.exists(csv_path):
                print(f"  [skip] {model_name}: not found")
                continue
            full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            full_df = add_indicators(full_df)
            full_df = full_df.dropna(subset=["sma20", "macd_hist"]).reset_index(drop=True)

            for use_filter in (False, True):
                strategy_name = "Filtered+Hysteresis" if use_filter else "Unfiltered"
                for slip in SLIPPAGE_SCENARIOS:
                    _, metrics = simulate(full_df, slip, use_filter=use_filter)
                    print(f"  {model_name:<6} {strategy_name:<20} slip={slip:.2f}%: "
                          f"final=${metrics['final_equity']:>9,.2f} ({metrics['total_return_pct']:+7.2f}%), "
                          f"trades={metrics['num_trades']:>4}, win_rate={metrics['win_rate_pct']:5.1f}%, "
                          f"buy_hold={metrics['buy_hold_return_pct']:+.2f}%")
                    all_rows.append({
                        "Interval": label, "Model": model_name, "Strategy": strategy_name,
                        "Slippage %": slip, **metrics,
                    })

    summary_df = pd.DataFrame(all_rows)
    summary_path = os.path.join(OUT_DIR, "pnl_compare_all_6mo_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
