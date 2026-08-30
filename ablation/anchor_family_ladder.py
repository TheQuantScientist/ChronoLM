"""Generate Anchor-family ablation ladder tables from a completed summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "anchor_results_family" / "anchor_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "anchor_results_family" / "ablation"

DATASET_ORDER = ["P12", "USHCN", "HumanActivity", "MIMIC"]
METHOD_ORDER = [
    "NaiveAnchor",
    "ExpoAnchor",
    "SparseAnchor",
    "ERMAnchor",
    "AutoAnchor",
]

METRIC_COLUMNS = {
    "global": ("MAE_scaled", "MSE_scaled"),
    "equal-variable": ("Equal_variable_MAE_scaled", "Equal_variable_MSE_scaled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Anchor-family ablation ladder from anchor_summary.csv. "
            "This script does not rerun forecasting experiments."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to anchor_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where ablation tables will be written.",
    )
    parser.add_argument(
        "--metric-mode",
        choices=sorted(METRIC_COLUMNS),
        default="global",
        help="Use APN-style global metrics or equal-variable averages.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")


def ordered_summary(summary_path: Path, metric_mode: str) -> pd.DataFrame:
    mae_col, mse_col = METRIC_COLUMNS[metric_mode]
    frame = pd.read_csv(summary_path)
    require_columns(
        frame,
        {"Dataset", "Method", "Predictions", "Fallback", "Time_s", mae_col, mse_col},
    )

    frame = frame.copy()
    frame["Method"] = pd.Categorical(frame["Method"], categories=METHOD_ORDER, ordered=True)
    frame["Dataset"] = pd.Categorical(frame["Dataset"], categories=DATASET_ORDER, ordered=True)
    frame = frame.dropna(subset=["Method", "Dataset"]).sort_values(["Dataset", "Method"])

    frame["MAE"] = frame[mae_col].astype(float)
    frame["MSE"] = frame[mse_col].astype(float)
    frame["Predictions"] = frame["Predictions"].astype(float).round().astype(int)
    frame["Fallback"] = frame["Fallback"].astype(float).round().astype(int)
    return frame


def second_distinct_mask(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        false_mask = pd.Series(False, index=values.index)
        return false_mask, false_mask

    unique_values = np.array(sorted(finite.unique()))
    best_value = unique_values[0]
    second_value = unique_values[1] if len(unique_values) > 1 else np.nan

    best = np.isclose(values.to_numpy(dtype=float), best_value, rtol=1e-12, atol=1e-12)
    if np.isnan(second_value):
        second = np.zeros(len(values), dtype=bool)
    else:
        second = np.isclose(values.to_numpy(dtype=float), second_value, rtol=1e-12, atol=1e-12)
    return pd.Series(best, index=values.index), pd.Series(second, index=values.index)


def add_ladder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    for dataset, group in frame.groupby("Dataset", sort=False, observed=True):
        group = group.copy().sort_values("Method")
        naive = group.loc[group["Method"] == "NaiveAnchor"]
        if naive.empty:
            raise ValueError(f"{dataset} is missing NaiveAnchor; cannot compute ladder deltas.")

        naive_mae = float(naive["MAE"].iloc[0])
        naive_mse = float(naive["MSE"].iloc[0])
        group["Delta_MAE_vs_Naive"] = naive_mae - group["MAE"]
        group["Delta_MSE_vs_Naive"] = naive_mse - group["MSE"]
        group["Pct_MAE_vs_Naive"] = 100.0 * group["Delta_MAE_vs_Naive"] / naive_mae
        group["Pct_MSE_vs_Naive"] = 100.0 * group["Delta_MSE_vs_Naive"] / naive_mse

        group["Delta_MAE_vs_Previous"] = group["MAE"].shift(1) - group["MAE"]
        group["Delta_MSE_vs_Previous"] = group["MSE"].shift(1) - group["MSE"]
        group.loc[group.index[0], ["Delta_MAE_vs_Previous", "Delta_MSE_vs_Previous"]] = 0.0

        group["MAE_dense_rank"] = group["MAE"].rank(method="dense", ascending=True).astype(int)
        group["MSE_dense_rank"] = group["MSE"].rank(method="dense", ascending=True).astype(int)
        group["MAE_best"], group["MAE_second"] = second_distinct_mask(group["MAE"])
        group["MSE_best"], group["MSE_second"] = second_distinct_mask(group["MSE"])
        outputs.append(group)

    return pd.concat(outputs, ignore_index=True)


def best_methods_text(group: pd.DataFrame, metric: str) -> str:
    best_value = group[metric].min()
    names = group.loc[np.isclose(group[metric], best_value), "Method"].astype(str).tolist()
    return "/".join(names)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in frame.groupby("Dataset", sort=False, observed=True):
        auto = group.loc[group["Method"] == "AutoAnchor"]
        naive = group.loc[group["Method"] == "NaiveAnchor"]
        if auto.empty or naive.empty:
            continue

        rows.append(
            {
                "Dataset": str(dataset),
                "Predictions": int(group["Predictions"].iloc[0]),
                "Best_MSE_Method": best_methods_text(group, "MSE"),
                "Best_MSE": float(group["MSE"].min()),
                "Best_MAE_Method": best_methods_text(group, "MAE"),
                "Best_MAE": float(group["MAE"].min()),
                "Auto_MSE": float(auto["MSE"].iloc[0]),
                "Auto_MAE": float(auto["MAE"].iloc[0]),
                "Auto_Delta_MSE_vs_Naive": float(auto["Delta_MSE_vs_Naive"].iloc[0]),
                "Auto_Delta_MAE_vs_Naive": float(auto["Delta_MAE_vs_Naive"].iloc[0]),
                "Auto_Pct_MSE_vs_Naive": float(auto["Pct_MSE_vs_Naive"].iloc[0]),
                "Auto_Pct_MAE_vs_Naive": float(auto["Pct_MAE_vs_Naive"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def fmt_float(value: float, digits: int = 4, signed: bool = False) -> str:
    if pd.isna(value):
        value = 0.0
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        value = 0.0
    return f"{value:+.1f}\\%"


def md_marker(value: float, best: bool, second: bool) -> str:
    text = fmt_float(value)
    if best:
        return f"**{text}**"
    if second:
        return f"<u>{text}</u>"
    return text


def tex_marker(value: float, best: bool, second: bool) -> str:
    text = fmt_float(value)
    if best:
        return f"\\textbf{{{text}}}"
    if second:
        return f"\\underline{{{text}}}"
    return text


def markdown_ladder(frame: pd.DataFrame, metric_mode: str) -> str:
    lines = [
        "# Anchor Family Ablation Ladder",
        "",
        f"Metric mode: `{metric_mode}`. Lower MAE/MSE is better. Positive deltas mean improvement over NaiveAnchor.",
        "",
        "| Dataset | Method | MSE | Delta MSE vs Naive | MAE | Delta MAE vs Naive |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Method']} | "
            f"{md_marker(row['MSE'], row['MSE_best'], row['MSE_second'])} | "
            f"{fmt_float(row['Delta_MSE_vs_Naive'], signed=True)} | "
            f"{md_marker(row['MAE'], row['MAE_best'], row['MAE_second'])} | "
            f"{fmt_float(row['Delta_MAE_vs_Naive'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "Best values within each dataset are bold; second-best values are underlined.",
            "Deltas are computed against NaiveAnchor on the same dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_ladder(frame: pd.DataFrame, metric_mode: str) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}",
        "\\begin{table*}[t]",
        "\\caption{Anchor-family ablation ladder. Lower is better for MSE and MAE.",
        "Bold marks the best Anchor-family value within each dataset; underline marks",
        "the second best. Positive deltas indicate improvement over NaiveAnchor.}",
        "\\label{tab:anchor-family-ladder}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & MSE & $\\Delta$ MSE vs Naive & MAE & $\\Delta$ MAE vs Naive \\\\",
        "\\midrule",
    ]

    dataset_groups = list(frame.groupby("Dataset", sort=False, observed=True))
    for dataset_index, (dataset, group) in enumerate(dataset_groups):
        if dataset_index > 0:
            lines.append("\\midrule")
        predictions = int(group["Predictions"].iloc[0])
        dataset_cell = (
            "\\multirow{5}{*}{\\begin{tabular}[c]{@{}c@{}}"
            f"{dataset}\\\\(predictions={predictions})"
            "\\end{tabular}}"
        )

        for row_index, (_, row) in enumerate(group.iterrows()):
            first_cell = dataset_cell if row_index == 0 else ""
            line = (
                f"{first_cell} & {row['Method']} & "
                f"{tex_marker(row['MSE'], row['MSE_best'], row['MSE_second'])} & "
                f"{fmt_float(row['Delta_MSE_vs_Naive'], signed=True)} & "
                f"{tex_marker(row['MAE'], row['MAE_best'], row['MAE_second'])} & "
                f"{fmt_float(row['Delta_MAE_vs_Naive'], signed=True)} \\\\"
            )
            lines.append(line)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            f"% Generated from {DEFAULT_SUMMARY.name} using {metric_mode} metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(frame: pd.DataFrame, output_dir: Path, metric_mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "anchor_family_ladder.csv"
    summary_path = output_dir / "anchor_family_ladder_summary.csv"
    md_path = output_dir / "anchor_family_ladder.md"
    tex_path = output_dir / "anchor_family_ladder.tex"

    csv_columns = [
        "Dataset",
        "Method",
        "Predictions",
        "Fallback",
        "MAE",
        "MSE",
        "Delta_MAE_vs_Naive",
        "Delta_MSE_vs_Naive",
        "Pct_MAE_vs_Naive",
        "Pct_MSE_vs_Naive",
        "Delta_MAE_vs_Previous",
        "Delta_MSE_vs_Previous",
        "MAE_dense_rank",
        "MSE_dense_rank",
        "MAE_best",
        "MAE_second",
        "MSE_best",
        "MSE_second",
    ]
    frame[csv_columns].to_csv(long_path, index=False)
    build_summary(frame).to_csv(summary_path, index=False)
    md_path.write_text(markdown_ladder(frame, metric_mode), encoding="utf-8")
    tex_path.write_text(latex_ladder(frame, metric_mode), encoding="utf-8")

    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


def main() -> None:
    args = parse_args()
    frame = ordered_summary(args.summary, args.metric_mode)
    frame = add_ladder_columns(frame)
    write_outputs(frame, args.output_dir, args.metric_mode)


if __name__ == "__main__":
    main()
