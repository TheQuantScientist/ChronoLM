# Anchor Family Ablations

Generate manuscript-ready ablation ladder tables from a completed Anchor-family
summary.

```bash
python ablation/anchor_family_ladder.py \
  --summary anchor_results_family/anchor_summary.csv \
  --output-dir anchor_results_family/ablation
```

The script does not rerun forecasting. It reads the existing summary CSV and
writes:

- `anchor_family_ladder.csv`: detailed long-form ablation table.
- `anchor_family_ladder_summary.csv`: best-method and AutoAnchor gain summary.
- `anchor_family_ladder.md`: readable Markdown table.
- `anchor_family_ladder.tex`: main-manuscript LaTeX table.

By default, the script uses APN-style global masked metrics. To use equal-variable
averages instead:

```bash
python ablation/anchor_family_ladder.py --metric-mode equal-variable
```
