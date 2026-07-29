"""Run APN paper baseline scripts from the repository root.

The scripts under APN/scripts are the source of truth for model settings. This
wrapper keeps their working directory as APN so relative paths, logs, and data
preparation follow the upstream project.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronolm.apn import APN_ROOT


PAPER_MODELS = [
    "PrimeNet",
    "NeuralFlows",
    "CRU",
    "mTAN",
    "SeFT",
    "GNeuralFlow",
    "GRU_D",
    "Raindrop",
    "Warpformer",
    "tPatchGNN",
    "GraFITi",
    "APN",
]

DATASETS = ["HumanActivity", "USHCN", "P12", "MIMIC_III"]

MODEL_ALIASES = {
    **{model.lower(): model for model in PAPER_MODELS},
    "gru-d": "GRU_D",
    "grud": "GRU_D",
    "tpatchgnn": "tPatchGNN",
    "mtan": "mTAN",
}

DATASET_ALIASES = {
    "humanactivity": "HumanActivity",
    "human_activity": "HumanActivity",
    "ushcn": "USHCN",
    "physionet": "P12",
    "physionet2012": "P12",
    "physionet_2012": "P12",
    "p12": "P12",
    "mimic": "MIMIC_III",
    "mimiciii": "MIMIC_III",
    "mimic_iii": "MIMIC_III",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run APN paper model scripts with upstream APN settings."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="APN script folder to run. May be repeated or comma separated. Default: paper table models.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset script to run. May be repeated or comma separated. Default: all four paper datasets.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the availability matrix and exit.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run scripts. Without this flag the wrapper only prints commands.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later scripts after one script fails.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("apn_model_runs"),
        help="Wrapper log directory. APN's own scripts still write APN/logs.",
    )
    return parser.parse_args()


def expand_selection(
    raw_values: list[str],
    defaults: list[str],
    aliases: dict[str, str],
) -> list[str]:
    if not raw_values:
        return defaults

    selected: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for item in raw_value.split(","):
            name = item.strip()
            if not name:
                continue
            if name.lower() == "all":
                for default in defaults:
                    if default not in seen:
                        seen.add(default)
                        selected.append(default)
                continue
            canonical = aliases.get(name.lower(), name)
            if canonical not in seen:
                seen.add(canonical)
                selected.append(canonical)
    return selected


def script_path(model: str, dataset: str) -> Path:
    return APN_ROOT / "scripts" / model / f"{dataset}.sh"


def print_matrix(models: list[str], datasets: list[str]) -> None:
    print("APN paper-script availability:")
    header = "Model".ljust(16) + " ".join(dataset.rjust(14) for dataset in datasets)
    print(header)
    print("-" * len(header))
    for model in models:
        cells = []
        for dataset in datasets:
            cells.append(("yes" if script_path(model, dataset).exists() else "missing").rjust(14))
        print(model.ljust(16) + " ".join(cells))


def run_script(model: str, dataset: str, log_root: Path) -> int:
    script = script_path(model, dataset)
    if not script.exists():
        print(f"[missing] {script}", file=sys.stderr)
        return 2

    APN_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    relative_script = script.relative_to(APN_ROOT)
    wrapper_log = log_root / f"{model}_{dataset}.log"

    command = ["bash", str(relative_script)]
    print(f"[run] cwd={APN_ROOT} {' '.join(command)}")
    start = time.time()
    with wrapper_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=APN_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    elapsed = time.time() - start
    status = "ok" if return_code == 0 else f"failed:{return_code}"
    print(f"[{status}] {model}/{dataset} in {elapsed:.1f}s, wrapper log: {wrapper_log}")
    return return_code


def main() -> None:
    args = parse_args()
    models = expand_selection(args.model, PAPER_MODELS, MODEL_ALIASES)
    datasets = expand_selection(args.dataset, DATASETS, DATASET_ALIASES)

    if args.list:
        print_matrix(models, datasets)
        return

    scripts = [(model, dataset) for model in models for dataset in datasets]
    if not args.execute:
        print("Dry run. Add --execute to run these APN scripts.")
        for model, dataset in scripts:
            script = script_path(model, dataset)
            status = "ok" if script.exists() else "missing"
            print(f"[{status}] cd {APN_ROOT} && bash {script.relative_to(APN_ROOT)}")
        return

    failures: list[tuple[str, str, int]] = []
    for model, dataset in scripts:
        return_code = run_script(model, dataset, args.log_root)
        if return_code != 0:
            failures.append((model, dataset, return_code))
            if not args.continue_on_error:
                break

    if failures:
        print("Failed scripts:", file=sys.stderr)
        for model, dataset, return_code in failures:
            print(f"  {model}/{dataset}: exit {return_code}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
