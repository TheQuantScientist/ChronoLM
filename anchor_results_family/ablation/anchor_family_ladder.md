# Anchor Family Ablation Ladder

Metric mode: `global`. Lower MAE/MSE is better. Positive deltas mean improvement over NaiveAnchor.

| Dataset | Method | MSE | Delta MSE vs Naive | MAE | Delta MAE vs Naive |
|---|---|---:|---:|---:|---:|
| P12 | NaiveAnchor | 0.4154 | +0.0000 | 0.3981 | +0.0000 |
| P12 | ExpoAnchor | <u>0.3204</u> | +0.0951 | <u>0.3716</u> | +0.0265 |
| P12 | SparseAnchor | 0.4154 | +0.0000 | 0.3981 | +0.0000 |
| P12 | ERMAnchor | **0.2962** | +0.1192 | **0.3556** | +0.0425 |
| P12 | AutoAnchor | **0.2962** | +0.1192 | **0.3556** | +0.0425 |
| USHCN | NaiveAnchor | 0.3640 | +0.0000 | 0.3291 | +0.0000 |
| USHCN | ExpoAnchor | 0.5923 | -0.2284 | 0.5117 | -0.1826 |
| USHCN | SparseAnchor | 0.2661 | +0.0978 | 0.2812 | +0.0479 |
| USHCN | ERMAnchor | <u>0.1662</u> | +0.1977 | <u>0.2635</u> | +0.0656 |
| USHCN | AutoAnchor | **0.1565** | +0.2074 | **0.2187** | +0.1104 |
| HumanActivity | NaiveAnchor | 0.0605 | +0.0000 | 0.1225 | +0.0000 |
| HumanActivity | ExpoAnchor | **0.0420** | +0.0185 | <u>0.1148</u> | +0.0078 |
| HumanActivity | SparseAnchor | 0.0605 | +0.0000 | 0.1225 | +0.0000 |
| HumanActivity | ERMAnchor | <u>0.0428</u> | +0.0177 | **0.1131** | +0.0094 |
| HumanActivity | AutoAnchor | **0.0420** | +0.0185 | <u>0.1148</u> | +0.0078 |

Best values within each dataset are bold; second-best values are underlined.
Deltas are computed against NaiveAnchor on the same dataset.
