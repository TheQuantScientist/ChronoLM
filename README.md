# ChronoLM

ChronoLM compares small language models against APN on irregular clinical time
series forecasting. The current benchmark uses the APN PhysioNet 2012 (P12)
data pipeline, split, scaling, and metrics, then replaces the trained APN model
with zero-shot LLM forecasts through an OpenAI-compatible API.

## Repository Layout

| Path | Purpose |
|------|---------|
| `APN/` | Upstream APN implementation and baseline training scripts. Keep this close to upstream. |
| `APN/data/` | APN data-loader source code and vendored `tsdm` code. This is source code, not raw data. |
| `src/chronolm/` | Active ChronoLM package code. |
| `run_p12_gemma.py` | Root-level entry point for the main Gemma-vs-APN experiment. |
| `scripts/` | Small diagnostic scripts for APN P12 data inspection. |
| `legacy/ollama_sliding_window/` | Older processed-CSV/Ollama prototype code, kept only for reference. |

Raw dataset cache lives outside the repo:

```bash
~/.tsdm/
```

## Setup

Python 3.11 is recommended because APN was tested against that version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r APN/requirements.txt
```

If the environment is missing common packages, install:

```bash
pip install torch pandas numpy scikit-learn requests pyarrow
```

## Configure The LLM Endpoint

The Gemma experiment reads these environment variables:

```bash
export CHRONOLM_API_URL="https://your-server/v1/chat/completions"
export CHRONOLM_MODEL_ID="google/gemma-3-4b-it"
export CHRONOLM_API_KEY="your-api-key"
```

Defaults are still present in the code for the current internal LiteLLM proxy,
but environment variables should be preferred for new runs.

## Run The Main Experiment

Run from the repository root:

```bash
python run_p12_gemma.py
```

The first run may download and preprocess PhysioNet 2012 into `~/.tsdm/`.
Later runs reuse the cache.

For a long run under `nohup`:

```bash
nohup python run_p12_gemma.py > gemma_run.log 2>&1 &
tail -f gemma_run.log
```

The runner prints compact `INPUT -> CALL -> OUTPUT -> PARSE/SCORE` traces for
valid forecast samples. By default it traces every valid sample. To reduce log
volume, increase `CHRONOLM_TRACE_EVERY`:

```bash
export CHRONOLM_TRACE_EVERY=10
export CHRONOLM_PROGRESS_EVERY=25
python run_p12_gemma.py
```

## Diagnostic Scripts

Run these from the repository root:

```bash
python scripts/p12_target_coverage.py
python scripts/p12_obs_count.py
python scripts/p12_debug_splits.py
```

## Outputs

The Gemma runner writes:

| File | Purpose |
|------|---------|
| `Gemma_temp06_P12_Results.csv` | Summary MAE/MSE by variable. |
| `Gemma_temp06_P12_Checkpoint.csv` | Per-variable checkpoint for resuming interrupted runs. |
| `Gemma_temp06_P12_DetailLog.csv` | Per-prediction actual vs predicted values. |
| `gemma_temp06_debug.log` | Prompt, response, fallback, and progress logging. |

Generated CSV and log files are ignored by git.

## APN Baseline

APN baseline training still uses the upstream scripts:

```bash
cd APN
chmod +x ./scripts/APN/P12.sh
./scripts/APN/P12.sh
```

That path is intentionally separate from the ChronoLM runner.
