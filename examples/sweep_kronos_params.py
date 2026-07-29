"""
Sampling-hyperparameter sweep for KronosPredictor.predict_batch on BTC/USDT
(daily granularity only, CPU-only friendly).

Reuses the data-fetching / block-building / metric logic from
examples/backtest_btc_accuracy.py so results are directly comparable to the
daily-granularity baseline reported in
examples/accuracy_backtest_output/findings.txt (context_len=90, block_len=5,
TEST_DAYS=182, i.e. 36 non-overlapping walk-forward blocks).

Stage 1 (coarse grid): T x top_p x sample_count over a small grid, one
predict_batch call per combo (all 36 blocks batched together -- cheap on CPU,
~5-20s/combo depending on sample_count).

Stage 2 (refinement): takes the best combo(s) from stage 1 by directional
accuracy and probes nearby T values plus sample_count=5.

Run with:
    .venv/bin/python examples/sweep_kronos_params.py
"""
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
from backtest_btc_accuracy import (  # noqa: E402
    fetch_klines, klines_to_df, build_blocks, compute_metrics,
)

SYMBOL = "BTCUSDT"
TEST_DAYS = 182  # matches the daily config in backtest_btc_accuracy.py for direct comparability
CONTEXT_LEN = 90
BLOCK_LEN = 5
MS_PER_BAR = 86_400_000
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_NAME = "NeoQuasar/Kronos-small"
OUT_DIR = os.path.join(os.path.dirname(__file__), "accuracy_backtest_output")
RESULTS_CSV = os.path.join(OUT_DIR, "param_sweep_results.csv")

# Stage 1: coarse grid
COARSE_T = [0.6, 0.8, 1.0, 1.2]
COARSE_TOP_P = [0.8, 0.9, 1.0]
COARSE_SAMPLE_COUNT = [1, 3]


def load_daily_blocks():
    num_blocks = TEST_DAYS // BLOCK_LEN
    total_needed = CONTEXT_LEN + num_blocks * BLOCK_LEN
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - int(total_needed * MS_PER_BAR * 1.3)

    raw = fetch_klines(SYMBOL, "1d", start_ms, end_ms)
    df = klines_to_df(raw)
    df = df.tail(total_needed).reset_index(drop=True)
    if len(df) < total_needed:
        raise RuntimeError(f"only got {len(df)} bars, needed {total_needed}")

    print(f"Loaded {len(df)} daily bars -> {num_blocks} walk-forward blocks "
          f"(context={CONTEXT_LEN}, block_len={BLOCK_LEN})")
    return build_blocks(df, CONTEXT_LEN, BLOCK_LEN, num_blocks)


def run_combo(predictor, df_list, xt_list, yt_list, actual_list, T, top_p, sample_count):
    t0 = time.time()
    pred_dfs = predictor.predict_batch(
        df_list, xt_list, yt_list, pred_len=BLOCK_LEN,
        T=T, top_p=top_p, sample_count=sample_count, verbose=False,
    )
    elapsed = time.time() - t0

    ts, actual, predicted, block_id = [], [], [], []
    for b, (y_ts, y_actual, pred_df) in enumerate(zip(yt_list, actual_list, pred_dfs)):
        ts.extend(list(y_ts))
        actual.extend(list(y_actual))
        predicted.extend(list(pred_df["close"].values))
        block_id.extend([b] * len(y_ts))

    result = pd.DataFrame(
        {"timestamp": ts, "actual": actual, "predicted": predicted, "block": block_id}
    ).sort_values("timestamp").reset_index(drop=True)
    metrics = compute_metrics(result)
    metrics.update({"T": T, "top_p": top_p, "sample_count": sample_count,
                     "elapsed_sec": round(elapsed, 2)})
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_NAME} + {TOKENIZER_NAME} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    df_list, xt_list, yt_list, actual_list = load_daily_blocks()

    all_rows = []
    sweep_start = time.time()

    print("\n=== Stage 1: coarse grid ===")
    combos = [
        (T, top_p, sc)
        for T in COARSE_T
        for top_p in COARSE_TOP_P
        for sc in COARSE_SAMPLE_COUNT
    ]
    for i, (T, top_p, sc) in enumerate(combos, 1):
        row = run_combo(predictor, df_list, xt_list, yt_list, actual_list, T, top_p, sc)
        all_rows.append(row)
        print(f"  [{i}/{len(combos)}] T={T} top_p={top_p} sample_count={sc} "
              f"-> DirAcc={row['DirectionalAccuracy']:.2f}% MAE={row['MAE']:.1f} "
              f"Corr={row['Correlation']:.3f} ({row['elapsed_sec']}s)")
        pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)

    stage1_df = pd.DataFrame(all_rows)
    top3 = stage1_df.sort_values(
        ["DirectionalAccuracy", "MAE"], ascending=[False, True]
    ).head(3)
    print("\nTop 3 from stage 1 (by directional accuracy, MAE tie-break):")
    print(top3.to_string(index=False))

    print("\n=== Stage 2: refinement around best combo(s) ===")
    best_T = top3.iloc[0]["T"]
    best_top_p = top3.iloc[0]["top_p"]
    refine_T = sorted({round(best_T - 0.1, 2), best_T, round(best_T + 0.1, 2)})
    refine_top_p = sorted({round(best_top_p - 0.05, 2), best_top_p,
                            min(1.0, round(best_top_p + 0.05, 2))})
    refine_sc = [1, 3, 5]

    refine_combos = [
        (T, top_p, sc)
        for T in refine_T
        for top_p in refine_top_p
        for sc in refine_sc
        if (T, top_p, sc) not in [(r["T"], r["top_p"], r["sample_count"]) for r in all_rows]
    ]
    for i, (T, top_p, sc) in enumerate(refine_combos, 1):
        row = run_combo(predictor, df_list, xt_list, yt_list, actual_list, T, top_p, sc)
        all_rows.append(row)
        print(f"  [{i}/{len(refine_combos)}] T={T} top_p={top_p} sample_count={sc} "
              f"-> DirAcc={row['DirectionalAccuracy']:.2f}% MAE={row['MAE']:.1f} "
              f"Corr={row['Correlation']:.3f} ({row['elapsed_sec']}s)")
        pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)

    total_elapsed = time.time() - sweep_start
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(RESULTS_CSV, index=False)

    overall_top5 = final_df.sort_values(
        ["DirectionalAccuracy", "MAE"], ascending=[False, True]
    ).head(5)
    print("\n" + "=" * 70)
    print(f"Sweep done in {total_elapsed / 60:.1f} min. Overall top 5 by directional accuracy:")
    print(overall_top5.to_string(index=False))
    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
