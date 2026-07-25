# Legacy Ollama Sliding-Window Prototype

This folder contains the older NIMAP/processed-CSV prototype code. It is not
the current APN/P12 benchmark path.

Known caveats:
- `run_experiment.py` still references a historical `src.*` package layout.
- Scripts expect files under `processed_data/extracted_features/`.
- Current runnable work is in `chronolm/experiments/p12_gemma.py`.
