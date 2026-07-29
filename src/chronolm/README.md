# ChronoLM Package

Active code outside upstream APN is intentionally small:

- `experiments/anchor_baseline.py`: anchor-only forecasting with APN data splits and metrics.
- `apn.py`: path helper for importing APN modules without changing directories.

Run from the repository root:

```bash
python run_anchor_baseline.py --dataset P12
```
