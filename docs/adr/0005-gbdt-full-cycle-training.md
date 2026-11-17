# ADR 0005: GBDT Training on Full Engine Cycles

## Status / ステータス

Accepted

採用

## Context / 背景

The `fd004_train_multiunit` dataset profile used `window_size=90`, meaning only the final
90 cycles of each training engine were included. This truncated training labels to y ∈ [0, 89].

The CMAPSS FD004 test set terminal RUL distribution (from `RUL_FD004.txt`) has:
- mean = 75.7, median = 85, range = [6, 125]
- 106 / 237 engines (44.7%) with terminal RUL > 89 — entirely outside the truncated train range
- 58 / 237 engines with terminal RUL = 125 (hard cap)

Consequence: the model had never seen labels ≥ 90 during training and could not extrapolate
to the test terminal distribution, producing `backtest_bias = -34.58` (systematic
underprediction of 34.6 cycles on average) and `backtest_rmse ≈ 46.4`.

その結果、学習時に 90 以上の label を見ていない model は test terminal distribution に外挿できず、
`backtest_bias = -34.58` という大きな負方向 bias と、`backtest_rmse ≈ 46.4` という悪化した精度を生んでいました。

Three options were evaluated:

**Option A — Train on all cycles (window_size=None)**
Use every cycle from every training engine. Labels span y ∈ [0, 125] and cover the full
test terminal distribution. Downside: y = 125 accounts for ~49% of rows (piecewise-linear cap),
creating label imbalance.

**Option B — Raise window_size to 150+**
Include more cycles per engine without going to full history. Partial improvement but still
truncates engines shorter than 150 cycles.

**Option C — Separate profile / env-var override**
Keep `fd004_train_multiunit` unchanged (used by the UI) and load full-cycle data exclusively
in the benchmark script via `build_fd004_payload(window_size=None)`.

## Decision / 判断

Adopt **Option C** (implemented as Option A data loading within the benchmark script).

**Option C** を採用します。ただし実装は benchmark script 内で Option A 相当の full-cycle data loading を行う形です。

The `fd004_train_multiunit` profile (`window_size=90`) is preserved because it is used by
the browser-based forecasting GUI as a sample payload; changing it would alter the UI
demonstration data. The benchmark script (`main()` in `build_fd004_benchmark_summary.py`)
now loads a separate full-cycle training set via `build_fd004_payload(split="train",
window_size=None)` labelled `train_profile: "fd004_full_cycles"` in the generated summary.

Label imbalance (y = 125 pileup) is addressed by the existing `rul_sample_weights(tau=40)`
function, which gives y = 0 samples 22.8× more weight than y = 125 samples.

## Consequences / 帰結

- `backtest_rmse` improved from 46.4 → 15.2 (−67%).
- `backtest_bias` improved from −34.6 → −2.2 (near-zero).
- `backtest_cov90` improved from 0.28 → 0.91 (target ≥ 0.90 achieved).
- `validation_rmse` (engine-holdout) reflects realistic out-of-distribution performance
  rather than in-sample time-series interpolation.
- The GBDT `fit_gbdt_pipeline` function now uses an engine-aware train/val split (GroupKFold
  equivalent: last 20% of engines held out) to prevent time-series autocorrelation leakage
  across the boundary.
- `low_rul_rmse_30` increased slightly (4.94 → 5.96) because the model must now share
  capacity across a wider label range; this is an acceptable trade-off given the test-set
  terminal RUL distribution.
- The GBDT benchmark default preset was changed from `"fast"` to `"full"` (`max_iter` up to
  600, 6 candidate configurations) to match the publication-quality target.

- `backtest_rmse` は 46.4 → 15.2 に改善した。
- `backtest_bias` は −34.6 → −2.2 に改善し、ほぼゼロに近づいた。
- `backtest_cov90` は 0.28 → 0.91 に改善し、目標の 0.90 を超えた。
- `validation_rmse` は、時系列の in-sample 補間ではなく、より現実的な engine-holdout 性能を表すようになった。
- benchmark の既定 preset は、publication-quality を前提に `fast` から `full` へ変更された。
