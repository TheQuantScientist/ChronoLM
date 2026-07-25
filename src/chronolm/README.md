# ChronoLM Package

This package contains active project code that is not part of upstream APN.

- `experiments/p12_gemma.py`: Gemma/SLM zero-shot forecasting on the APN P12 task.
- `tools/`: small APN P12 inspection modules used while validating the experiment.
- `apn.py`: path helper that allows APN imports without changing directories.

Run the main experiment from the repository root:

```bash
python run_p12_gemma.py
```
