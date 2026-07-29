"""Meta-labeling classifier for Kronos rule-based trade filter.

Trains a shallow gradient-boosted classifier on the entry-time features of
historical trades (produced by run_walkforward.py / run_walkforward_1h.py)
to predict whether a trade will win or lose. This does not replace the
SMA/MACD/hysteresis rule filter -- it's a diagnostic + optional secondary
filter layered on top of its output, to explain *which* entry conditions the
strategy tends to get wrong.

Usage:
    .venv/bin/python examples/rl_trading/train_meta_label.py \
        --trade-log examples/rl_trading/output/all_coins_trade_log.csv \
        --output-dir examples/rl_trading/output/meta_label/daily
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "predicted_move_pct_at_entry",
    "sma_dist_pct_at_entry",
    "macd_hist_at_entry",
    "macd_hist_slope_at_entry",
    "direction",
]
TARGET_COL = "outcome"


def load_data(trade_log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trade_log_path, parse_dates=["entry_timestamp", "exit_timestamp"])
    df = df.sort_values("entry_timestamp").reset_index(drop=True)
    # A trade's macd_hist_slope is NaN only when it's a symbol's first trade
    # (no prior bar to diff against). Fill with 0, i.e. "no slope information
    # / treat as flat", rather than dropping the row or imputing from other
    # symbols' data (which would leak cross-symbol information).
    df["macd_hist_slope_at_entry"] = df["macd_hist_slope_at_entry"].fillna(0.0)
    return df


def chronological_split(
    df: pd.DataFrame, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
    return model


def evaluate_model(
    model: XGBClassifier, test_df: pd.DataFrame
) -> dict:
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    baseline_win_rate = y_test.mean()

    metrics = {
        "n_test_trades": int(len(test_df)),
        "baseline_win_rate": float(baseline_win_rate),
        "model_accuracy": float(accuracy_score(y_test, y_pred)),
        "model_precision": float(precision),
        "model_recall": float(recall),
        "model_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def threshold_backtest(
    model: XGBClassifier, test_df: pd.DataFrame, thresholds: list[float]
) -> pd.DataFrame:
    X_test = test_df[FEATURE_COLS]
    proba = model.predict_proba(X_test)[:, 1]
    test_df = test_df.assign(win_proba=proba)

    rows = []
    for thresh in thresholds:
        selected = test_df[test_df["win_proba"] >= thresh]
        n = len(selected)
        win_rate = selected[TARGET_COL].mean() if n > 0 else float("nan")
        # Simple additive sum of realized_return_pct across taken trades --
        # a diagnostic proxy for relative performance, not a compounded
        # equity curve (no position sizing model here).
        total_return = selected["realized_return_pct"].sum() if n > 0 else 0.0
        rows.append(
            {
                "threshold": thresh,
                "num_trades": n,
                "pct_of_test_trades_kept": n / len(test_df) if len(test_df) else 0.0,
                "win_rate": win_rate,
                "total_return_pct": total_return,
            }
        )
    return pd.DataFrame(rows)


def shap_analysis(
    model: XGBClassifier, test_df: pd.DataFrame, output_dir: Path
) -> dict:
    X_test = test_df[FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Full-test-set summary plot (standard SHAP usage).
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_all_test_trades.png", dpi=150)
    plt.close()

    # Restrict to trades that actually lost, to see which feature values
    # were pushing the model's (log-odds) output toward the losing class.
    loss_mask = (test_df[TARGET_COL] == 0).to_numpy()
    loss_shap_values = shap_values.values[loss_mask]
    loss_X = X_test[loss_mask]

    if loss_mask.sum() > 0:
        plt.figure()
        shap.summary_plot(loss_shap_values, loss_X, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_losing_test_trades.png", dpi=150)
        plt.close()

        mean_shap_on_losses = pd.Series(
            loss_shap_values.mean(axis=0), index=FEATURE_COLS
        ).sort_values()
    else:
        mean_shap_on_losses = pd.Series(dtype=float)

    return {"mean_shap_on_losing_trades": mean_shap_on_losses.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.trade_log)
    train_df, test_df = chronological_split(df, args.test_frac)
    print(f"Loaded {len(df)} trades from {args.trade_log}")
    print(
        f"Chronological split: {len(train_df)} train "
        f"({train_df['entry_timestamp'].min()} -> {train_df['entry_timestamp'].max()}), "
        f"{len(test_df)} test "
        f"({test_df['entry_timestamp'].min()} -> {test_df['entry_timestamp'].max()})"
    )

    model = train_model(train_df)
    joblib.dump(model, args.output_dir / "meta_label_model.joblib")

    metrics = evaluate_model(model, test_df)
    print("\n=== Held-out test set metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    thresholds = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    backtest_df = threshold_backtest(model, test_df, thresholds)
    backtest_df.to_csv(args.output_dir / "threshold_backtest.csv", index=False)
    print("\n=== Threshold backtest (test set) ===")
    print(backtest_df.to_string(index=False))

    shap_result = shap_analysis(model, test_df, args.output_dir)
    print("\n=== Mean SHAP value per feature, on actually-losing test trades ===")
    print("(more negative = pushes model harder toward predicting a loss)")
    for feat, val in shap_result["mean_shap_on_losing_trades"].items():
        print(f"  {feat}: {val:.4f}")

    summary = {
        "metrics": metrics,
        "threshold_backtest": backtest_df.to_dict(orient="records"),
        "shap": shap_result,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plain-English summary of top loss-associated conditions.
    ranked = sorted(
        shap_result["mean_shap_on_losing_trades"].items(), key=lambda kv: kv[1]
    )
    print("\n=== Plain-English summary ===")
    print(
        f"Baseline (take-every-trade) win rate on held-out test set: "
        f"{metrics['baseline_win_rate']:.1%} ({metrics['n_test_trades']} trades)."
    )
    print(
        f"Meta-model accuracy: {metrics['model_accuracy']:.1%}, "
        f"ROC-AUC: {metrics['model_roc_auc']:.3f}."
    )
    print("Top entry conditions most associated with the strategy's losing trades:")
    for feat, val in ranked[:3]:
        print(f"  - {feat} (mean SHAP contribution on losses: {val:.4f})")

    no_filter_row = backtest_df[backtest_df["threshold"] == 0.0].iloc[0]
    best_row = backtest_df.iloc[backtest_df["win_rate"].idxmax()]
    print(
        f"Taking every filtered trade (threshold=0): win_rate="
        f"{no_filter_row['win_rate']:.1%}, total_return={no_filter_row['total_return_pct']:.2f}%, "
        f"n={int(no_filter_row['num_trades'])}."
    )
    print(
        f"Best win-rate threshold in sweep ({best_row['threshold']}): win_rate="
        f"{best_row['win_rate']:.1%}, total_return={best_row['total_return_pct']:.2f}%, "
        f"n={int(best_row['num_trades'])}."
    )


if __name__ == "__main__":
    main()
