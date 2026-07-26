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
| `run_p12_llm.py` | Root-level entry point for the main LLM-vs-APN experiment. |
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

The experiment reads these environment variables:

```bash
export CHRONOLM_API_URL="https://your-server/v1/chat/completions"
export CHRONOLM_MODEL_ID="Qwen/Qwen3.5-4B"
export CHRONOLM_API_KEY="your-api-key"
export CHRONOLM_ENABLE_THINKING=false
```

Defaults are still present in the code for the current internal LiteLLM proxy
and Qwen model, but environment variables should be preferred for model sweeps.
Qwen thinking mode is off by default for cleaner numeric forecasting output.

## Run The Main Experiment

Run from the repository root:

```bash
python run_p12_llm.py
```

The first run may download and preprocess PhysioNet 2012 into `~/.tsdm/`.
Later runs reuse the cache.

For a long run under `nohup`:

```bash
nohup python run_p12_llm.py > llm_run.log 2>&1 &
tail -f llm_run.log
```

The runner prints compact `INPUT -> CALL -> OUTPUT -> PARSE/SCORE` traces for
valid forecast samples. By default it traces every valid sample. To reduce log
volume, increase `CHRONOLM_TRACE_EVERY`:

```bash
export CHRONOLM_TRACE_EVERY=10
export CHRONOLM_PROGRESS_EVERY=25
export CHRONOLM_MAX_RETRIES=6
python run_p12_llm.py
```

To run a different model, set `CHRONOLM_MODEL_ID`. Output filenames are derived
from the model id by default, or from `CHRONOLM_RUN_NAME` when provided.

```bash
export CHRONOLM_MODEL_ID="Qwen/Qwen3.5-4B"
export CHRONOLM_RUN_NAME="qwen3_5_4b_temp06"
export CHRONOLM_ENABLE_THINKING=false
nohup python run_p12_llm.py > qwen3_5_4b_run.log 2>&1 &
```

For OpenAI GPT-5 mini, use the dedicated launcher. It reads the key from
`OPENAI_API_KEY` in the shell or from a local `.env` file, and defaults to
`reasoning_effort=minimal`, low verbosity, and a 64-token output cap for short
numeric outputs. GPT-5 mini currently rejects non-default temperature values, so
the launcher omits temperature. It also uses a GPT-only clinical prompt focused
on conditional expected values to reduce squared-error blowups.

```bash
export OPENAI_API_KEY="your-openai-key"
nohup python run_p12_gpt5_mini.py > gpt5_mini_run.log 2>&1 &
tail -f gpt5_mini_run.log
```

## Diagnostic Scripts

Run these from the repository root:

```bash
python scripts/p12_target_coverage.py
python scripts/p12_obs_count.py
python scripts/p12_debug_splits.py
```

## Outputs

The runner writes:

| File | Purpose |
|------|---------|
| `<run_name>_P12_Results.csv` | Summary MAE/MSE by variable. |
| `<run_name>_P12_Checkpoint.csv` | Per-variable checkpoint for resuming interrupted runs. |
| `<run_name>_P12_DetailLog.csv` | Per-prediction actual vs predicted values. |
| `<run_name>_debug.log` | Prompt, response, fallback, and progress logging. |

Generated CSV and log files are ignored by git.

## APN Baseline

APN baseline training still uses the upstream scripts:

```bash
cd APN
chmod +x ./scripts/APN/P12.sh
./scripts/APN/P12.sh
```

That path is intentionally separate from the ChronoLM runner.
