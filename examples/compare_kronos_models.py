"""
Compare Kronos-small vs. Kronos-base on the same BTC/USDT walk-forward
accuracy backtest.

Reads the *_results.csv files each model run already produced:
  - Kronos-small: examples/accuracy_backtest_output/btc_{interval}_results.csv
  - Kronos-base:  examples/accuracy_backtest_output/kronos-base/btc_{interval}_results.csv
(both produced by backtest_btc_accuracy.py -- see KRONOS_MODEL_NAME /
KRONOS_OUT_SUBDIR env vars in that script for how the base run was kept
separate from the small run's files.)

Run after both backtests have completed:
    .venv/bin/python examples/compare_kronos_models.py
"""
import os
import pandas as pd

from backtest_btc_accuracy import compute_metrics  # reuse the exact same metric definitions

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


def main():
    rows = []
    for model_name, model_dir in MODELS.items():
        for interval, label in INTERVALS.items():
            csv_path = os.path.join(model_dir, f"btc_{interval}_results.csv")
            if not os.path.exists(csv_path):
                print(f"[skip] {model_name} / {label}: {csv_path} not found")
                continue
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            metrics = compute_metrics(df)
            rows.append({"Model": model_name, "Granularity": label, **metrics})

    if not rows:
        print("No results found for either model -- nothing to compare.")
        return

    result = pd.DataFrame(rows)
    result = result.sort_values(["Granularity", "Model"]).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("Kronos-small vs. Kronos-base — Accuracy Comparison (same BTC/USDT walk-forward setup)")
    print("=" * 100)
    print(result.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    out_path = os.path.join(OUT_DIR, "model_comparison_small_vs_base.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n--- Directional accuracy delta (base - small), positive = base is better ---")
    for interval_label in INTERVALS.values():
        sub = result[result["Granularity"] == interval_label]
        if len(sub) != 2:
            continue
        small_da = sub[sub["Model"] == "Kronos-small"]["DirectionalAccuracy"].iloc[0]
        base_da = sub[sub["Model"] == "Kronos-base"]["DirectionalAccuracy"].iloc[0]
        print(f"  {interval_label}: small={small_da:.2f}%  base={base_da:.2f}%  delta={base_da - small_da:+.2f}pp")


if __name__ == "__main__":
    main()
