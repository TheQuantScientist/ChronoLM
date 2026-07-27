# ChronoLM Package

This package contains active project code that is not part of upstream APN.

- `experiments/apn_llm.py`: LLM zero-shot forecasting on supported APN benchmark tasks.
- `tools/`: small APN P12 inspection modules used while validating the experiment.
- `apn.py`: path helper that allows APN imports without changing directories.

Run the main experiment from the repository root:

```bash
python run_apn_llm.py
```
