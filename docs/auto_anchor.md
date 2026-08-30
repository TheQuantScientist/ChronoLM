# AutoAnchor Method Note

AutoAnchor is the headline member of a deterministic anchor family for
irregular multivariate time series. The family asks whether simple history
anchors can match or beat heavier neural IMTS models under the exact APN
benchmark splits and scaled metrics.

## Anchor Family

| Method | Role |
|---|---|
| `NaiveAnchor` | Repeats the last observed value. |
| `ExpoAnchor` | Uses a fixed EMA anchor with alpha derived from the forecast/lookback ratio. |
| `SparseAnchor` | Uses sparse-history mode/persistence rules; falls back to last-value when no sparse baseline is detected. |
| `ERMAnchor` | Selects the best candidate by pooled APN train+validation scaled MSE. |
| `AutoAnchor` | Starts from ERM selection, then applies a generic history-structure prior when the observed history has a strong sparse, long-window, or seasonal signature. |

## Current Algorithm

1. Use APN data loaders, train/validation/test splits, lookback windows, and
   prediction horizons.
2. Build one candidate library for every dataset and variable. The library
   contains last-value, recent means, trimmed means, EMA anchors, local linear
   trends, phase-nearest anchors, and fixed convex mixtures of those components.
3. For `ERMAnchor` and `AutoAnchor`, select from the shared candidate library
   without using test labels:
   - `non_test_empirical_risk`: minimize pooled APN train+validation scaled
     MSE and optionally fit one scalar shrinkage coefficient.
   - `history_structural_prior`: when histories show a generic structural
     signature, select the corresponding candidate from observed history only.
4. Evaluate once on the APN test split and report APN-style global masked
   MAE/MSE from the detail log.

Dataset-specific code is limited to loading the APN task objects and providing
the public protocol metadata (`seq_len`, `pred_len`, columns, scaling).
AutoAnchor selection does not branch on dataset or variable names.

## Current Public Results

| Dataset | MAE | MSE |
|---|---:|---:|
| PhysioNet P12 | 0.3556 | 0.2962 |
| USHCN | 0.2187 | 0.1565 |
| HumanActivity | 0.1148 | 0.0420 |

These are single-run APN-style global masked metrics from
`anchor_results/anchor_summary.csv`.

## ICLR-Grade Checklist

- Lock the AutoAnchor algorithm before running MIMIC.
- Run and report the full anchor family.
- Add ablations for no shrinkage, no phase anchors, no trend anchors, and
  candidate-library size.
- Report per-dataset and pooled rank against APN, GraFITi, tPatchGNN,
  Warpformer, and classical anchors such as last-value and EMA.
- Include a no-test-tuning audit: record all candidate choices in calibration
  CSVs and keep test detail logs immutable after each run.
- Run seeds/splits whenever the upstream APN protocol exposes them; otherwise
  state that APN fixed split 0 is used.
