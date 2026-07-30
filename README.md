# ChronoLM

ChronoLM is now a compact benchmark workspace for asking a sharper question:
how far can simple history anchors go on irregular multivariate time-series
forecasting when evaluated with the same APN data splits and scaled metrics?

The upstream APN repository is kept under `APN/` and remains the source of truth
for trained APN and paper-baseline model settings. ChronoLM code outside that
folder only contains our AutoAnchor runner and orchestration utilities.

## Layout

| Path | Purpose |
|---|---|
| `APN/` | Upstream APN implementation, datasets, configs, and model scripts. |
| `src/chronolm/experiments/anchor_baseline.py` | AutoAnchor runner using APN data loaders and metrics. |
| `run_anchor_baseline.py` | Root-level launcher for anchor runs on P12, USHCN, and HumanActivity. |
| `run_apn_paper_models.py` | Root-level wrapper for APN paper model scripts. |
| `scripts/compute_global_metrics.py` | Utility for APN-style global MAE/MSE from detail logs. |
| `docs/auto_anchor.md` | Method note, current results, and ICLR-grade ablation checklist. |
| `apn_benchmark_results.md` | Current paper-table results with our AutoAnchor row. |

Generated logs, CSVs, checkpoints, and local environments are ignored by git.

## Setup

APN recommends Python 3.11.13 and PyTorch 2.6.0+cu124.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r APN/requirements.txt
pip install -e .
```

Public datasets are prepared by APN on first use:

| Dataset | Cache |
|---|---|
| P12 / PhysioNet 2012 | `~/.tsdm/` |
| USHCN | `~/.tsdm/` |
| HumanActivity | `APN/storage/datasets/HumanActivity` |

MIMIC requires credentialed access. Follow `APN/README.md` and place
`complete_tensor.csv` under `~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/`.

## Run Anchor Baselines

Run all currently supported anchor datasets:

```bash
python run_anchor_baseline.py --all
```

Run one dataset:

```bash
python run_anchor_baseline.py --dataset P12
python run_anchor_baseline.py --dataset USHCN
python run_anchor_baseline.py --dataset HumanActivity
```

Cheap smoke test:

```bash
python run_anchor_baseline.py --dataset USHCN --max-test-samples 20 --trace-every 5
```

Outputs go under `anchor_results/<dataset>/` and include:

| File | Purpose |
|---|---|
| `*_Calibration.csv` | Per-variable anchor method, fitted shrinkage, source, and rationale. |
| `*_Results.csv` | Per-variable scaled MAE/MSE. |
| `*_Checkpoint.csv` | Per-variable resume checkpoint. |
| `*_DetailLog.csv` | Per-target actual, prediction, and anchor values. |
| `*_debug.log` | Progress and compact trace logging. |
| `anchor_results/anchor_summary.csv` | APN-style global summary across completed datasets. |

AutoAnchor uses one shared candidate library for every supported dataset and
variable. Dataset-specific code is limited to APN data loading, split choice,
and benchmark window sizes. The selector first evaluates candidates on APN
train+validation targets; when the observed histories have a strong generic
signature, such as sparse mode dominance, long-window short-horizon dynamics,
or seasonal phase structure, a deterministic history-only prior selects the
corresponding candidate. Test labels are used only once for final reporting.
The calibration CSV records the source and rationale for every variable.

Compute APN-style metrics from any detail log:

```bash
python scripts/compute_global_metrics.py anchor_results/p12/auto_anchor_unified_p12_P12_DetailLog.csv
```

## Run APN Paper Models

List the APN paper-script matrix:

```bash
python run_apn_paper_models.py --list
```

Dry-run the exact APN commands from the root:

```bash
python run_apn_paper_models.py --model APN --dataset P12
```

Execute a script:

```bash
python run_apn_paper_models.py --model APN --dataset P12 --execute
```

Run every paper-table model on the public datasets:

```bash
python run_apn_paper_models.py \
  --dataset HumanActivity,USHCN,P12 \
  --execute \
  --continue-on-error
```

The wrapper does not rewrite APN hyperparameters or GPU ids. It runs each
`APN/scripts/<model>/<dataset>.sh` with `cwd=APN`, creates `APN/logs/`, and
writes wrapper logs under `apn_model_runs/`.
