"""Run ChronoLM's anchor-family baselines on APN benchmark splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronolm.experiments.anchor_baseline import (
    ANCHOR_FAMILY_METHODS,
    ANCHOR_METHOD_SLUGS,
    RUN_NAME_PREFIX,
    AnchorConfig,
    canonical_anchor_method,
    canonical_dataset_name,
    run,
)


DATASET_ORDER = ["P12", "USHCN", "HumanActivity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anchor-family baselines with APN data splits and metrics."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to run: P12, USHCN, HumanActivity, or all. May be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all currently supported anchor datasets.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Anchor method to run: NaiveAnchor, ExpoAnchor, SparseAnchor, "
            "ERMAnchor, AutoAnchor, or family. May be repeated."
        ),
    )
    parser.add_argument(
        "--family",
        action="store_true",
        help="Run the full anchor family: NaiveAnchor, ExpoAnchor, SparseAnchor, ERMAnchor, AutoAnchor.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("anchor_results"),
        help="Directory for result CSVs and logs.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit on test samples per dataset.",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=0,
        help="Log per-sample input/anchor/output every N valid samples. 0 disables traces.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log variable progress every N scanned samples.",
    )
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[str]:
    raw_names = args.dataset or []
    if args.all or any(name.lower() == "all" for name in raw_names):
        return DATASET_ORDER

    if not raw_names:
        return DATASET_ORDER

    seen: set[str] = set()
    datasets: list[str] = []
    for raw_name in raw_names:
        for part in raw_name.split(","):
            dataset = canonical_dataset_name(part)
            if dataset not in seen:
                seen.add(dataset)
                datasets.append(dataset)
    return datasets


def selected_methods(args: argparse.Namespace) -> list[str]:
    raw_names = args.method or []
    if args.family or any(name.lower() in {"all", "family"} for name in raw_names):
        return list(ANCHOR_FAMILY_METHODS)

    if not raw_names:
        return ["AutoAnchor"]

    seen: set[str] = set()
    methods: list[str] = []
    for raw_name in raw_names:
        for part in raw_name.split(","):
            method = canonical_anchor_method(part)
            if method not in seen:
                seen.add(method)
                methods.append(method)
    return methods


def main() -> None:
    args = parse_args()
    summaries: list[dict] = []

    for dataset in selected_datasets(args):
        for method in selected_methods(args):
            output_dir = args.output_root / dataset.lower()
            config = AnchorConfig(
                dataset_name=dataset,
                anchor_method=method,
                max_test_samples=args.max_test_samples,
                run_name=f"{RUN_NAME_PREFIX}_{ANCHOR_METHOD_SLUGS[method]}_{dataset.lower()}",
                output_dir=output_dir,
                trace_every=args.trace_every,
                progress_every=args.progress_every,
            )
            summaries.append(run(config))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "anchor_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Anchor summary written to {summary_path}")


if __name__ == "__main__":
    main()
