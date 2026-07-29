"""
Simple directional P&L simulation on top of the Kronos accuracy backtest.

Strategy (as simple as it gets): at each bar, look at what Kronos predicted
for this bar vs. the last REAL price you actually knew. If it predicted
higher -> go long for that bar. If lower -> go short. Apply that bar's real
return, then subtract a slippage/fee cost every time the position changes.

Run after backtest_btc_accuracy.py has produced the *_results.csv files:
    .venv/bin/python examples/pnl_backtest.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_SCENARIOS = [0.20, 0.54]  # percent per side, per user request

INTERVALS = {
    "1d": "Daily",
    "1h": "Hourly",
    "15m": "15-Minute",
}


def simulate(df, slippage_pct):
    df = df.copy()
    df["prev_actual"] = df["actual"].shift(1)
    df = df.dropna(subset=["prev_actual"]).reset_index(drop=True)

    pred_ret = (df["predicted"] - df["prev_actual"]) / df["prev_actual"]
    df["position"] = np.where(pred_ret > 0, 1, np.where(pred_ret < 0, -1, 0))
    df["bar_return"] = df["position"] * (df["actual"] / df["prev_actual"] - 1)

    prev_pos = df["position"].shift(1).fillna(0)
    position_change = (df["position"] - prev_pos).abs()  # 0, 1, or 2 (flip)
    df["slippage_cost"] = position_change * (slippage_pct / 100.0)
    df["net_return"] = df["bar_return"] - df["slippage_cost"]

    df["equity"] = INITIAL_CAPITAL * (1 + df["net_return"]).cumprod()
    bh_return = df["actual"] / df["prev_actual"] - 1
    df["buy_hold_equity"] = INITIAL_CAPITAL * (1 + bh_return).cumprod()

    num_trades = int((position_change > 0).sum())
    winning_bars = int((df["bar_return"] > 0).sum())
    active_bars = int((df["position"] != 0).sum())
    total_slippage_cost = float((df["slippage_cost"] * INITIAL_CAPITAL).sum())  # rough $ estimate, ignores compounding

    return df, {
        "final_equity": float(df["equity"].iloc[-1]),
        "total_return_pct": float((df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "buy_hold_final_equity": float(df["buy_hold_equity"].iloc[-1]),
        "buy_hold_return_pct": float((df["buy_hold_equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "num_trades": num_trades,
        "win_rate_pct": 100.0 * winning_bars / active_bars if active_bars else float("nan"),
        "total_slippage_cost_usd": total_slippage_cost,
    }


def plot_equity(interval, label, curves):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, df in curves.items():
        ax.plot(df["timestamp"], df["equity"], label=name, linewidth=1.5)
    # buy & hold is identical across slippage scenarios, plot once
    any_df = next(iter(curves.values()))
    ax.plot(any_df["timestamp"], any_df["buy_hold_equity"], label="Buy & Hold (no trading)",
            color="gray", linestyle="--", linewidth=1.3)
    ax.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"BTC/USDT {label} — Kronos Directional Strategy P&L (start ${INITIAL_CAPITAL:,.0f})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Account Value (USDT)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"btc_{interval}_pnl.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    all_summary = {}
    for interval, label in INTERVALS.items():
        csv_path = os.path.join(OUT_DIR, f"btc_{interval}_results.csv")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        curves = {}
        print(f"\n=== {label} ({interval}) ===")
        for slip in SLIPPAGE_SCENARIOS:
            sim_df, metrics = simulate(df, slip)
            curves[f"Strategy ({slip:.2f}% slippage)"] = sim_df
            all_summary[(label, slip)] = metrics
            print(f"  Slippage {slip:.2f}%: final=${metrics['final_equity']:,.2f} "
                  f"({metrics['total_return_pct']:+.2f}%), trades={metrics['num_trades']}, "
                  f"win_rate={metrics['win_rate_pct']:.1f}%, "
                  f"slippage_cost=${metrics['total_slippage_cost_usd']:,.2f}")
        print(f"  Buy & Hold: final=${metrics['buy_hold_final_equity']:,.2f} "
              f"({metrics['buy_hold_return_pct']:+.2f}%)")

        path = plot_equity(interval, label, curves)
        print(f"  Saved chart: {path}")

    print("\n" + "=" * 70)
    print(f"Summary (starting capital ${INITIAL_CAPITAL:,.0f} each)")
    print("=" * 70)
    rows = []
    for (label, slip), m in all_summary.items():
        rows.append({
            "Interval": label, "Slippage %": slip,
            "Final $": round(m["final_equity"], 2), "Return %": round(m["total_return_pct"], 2),
            "Trades": m["num_trades"], "Win Rate %": round(m["win_rate_pct"], 1),
            "Buy&Hold Return %": round(m["buy_hold_return_pct"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
