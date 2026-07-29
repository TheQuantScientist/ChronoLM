"""Repository-root launcher for P12 GPT-5 mini ablation experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chronolm.env import load_dotenv  # noqa: E402
from chronolm.experiments.apn_llm import (  # noqa: E402
    ExperimentConfig,
    compute_global_scaled_metrics,
    run,
)


ABLATION_SETTINGS = {
    "anchor_only": {
        "prompt_style": "gpt_clinical_calibrated_anchor_mse",
        "anchor_weight": "1.00",
    },
    "gpt_only": {
        "prompt_style": "gpt_clinical_mse",
        "anchor_weight": "0.00",
    },
    "anchor_gpt": {
        "prompt_style": "gpt_clinical_calibrated_anchor_mse",
        "anchor_weight": "0.95",
    },
    "anchor_gpt_no_context": {
        "prompt_style": "gpt_clinical_calibrated_anchor_mse",
        "anchor_weight": "0.95",
    },
    "anchor_gpt_generic_prompt": {
        "prompt_style": "gpt_clinical_calibrated_anchor_mse",
        "anchor_weight": "0.95",
    },
}


def mode_output_dir(mode: str, max_test_samples: int | None = None) -> Path:
    root = REPO_ROOT / "ablation_results" / "p12"
    if max_test_samples is not None:
        return root / "smoke" / f"{mode}_samples_{max_test_samples}"
    return root / mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P12 ablations for anchor-conditioned GPT-5 mini forecasting."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(ABLATION_SETTINGS),
        default="anchor_only",
        help="Ablation mode to run. Defaults to the cheap anchor-only ablation.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all ablations sequentially. This is expensive for GPT modes.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional smoke-test sample cap, forwarded to CHRONOLM_MAX_TEST_SAMPLES.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configurations without running the experiments.",
    )
    return parser.parse_args()


def configure_common_environment() -> None:
    load_dotenv(REPO_ROOT / ".env")

    os.environ["CHRONOLM_DATASET"] = "P12"
    os.environ["CHRONOLM_API_PROVIDER"] = "openrouter"
    os.environ["CHRONOLM_API_URL"] = "https://openrouter.ai/api/v1/chat/completions"
    os.environ["CHRONOLM_MODEL_ID"] = "openai/gpt-5-mini"
    os.environ["CHRONOLM_REASONING_EFFORT"] = "minimal"
    os.environ["CHRONOLM_INCLUDE_TEMPERATURE"] = "false"
    os.environ["CHRONOLM_MAX_TOKENS"] = "64"

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        os.environ["CHRONOLM_API_KEY"] = openrouter_api_key


def configure_mode(mode: str, max_test_samples: int | None) -> None:
    settings = ABLATION_SETTINGS[mode]
    run_name = f"gpt5_mini_p12_ablation_{mode}"
    output_dir = mode_output_dir(mode, max_test_samples)

    os.environ["CHRONOLM_ABLATION_MODE"] = mode
    os.environ["CHRONOLM_RUN_NAME"] = run_name
    os.environ["CHRONOLM_PROMPT_STYLE"] = settings["prompt_style"]
    os.environ["CHRONOLM_ANCHOR_BLEND_WEIGHT"] = settings["anchor_weight"]
    os.environ["CHRONOLM_OUTPUT_DIR"] = str(output_dir)

    if max_test_samples is None:
        os.environ.pop("CHRONOLM_MAX_TEST_SAMPLES", None)
    else:
        os.environ["CHRONOLM_MAX_TEST_SAMPLES"] = str(max_test_samples)


def require_api_key_if_needed(mode: str) -> None:
    if mode == "anchor_only":
        return

    if not os.getenv("CHRONOLM_API_KEY"):
        raise RuntimeError(
            "Set OPENROUTER_API_KEY before running GPT ablation modes. "
            "Anchor-only can run without an API key."
        )


def print_config(config: ExperimentConfig) -> None:
    print(f"mode={config.ablation_mode}")
    print(f"run_name={config.run_name}")
    print(f"prompt_style={config.prompt_style}")
    print(f"anchor_blend_weight={config.anchor_blend_weight}")
    print(f"output_csv={config.output_csv}")
    print(f"detail_log_csv={config.detail_log_csv}")
    print(f"debug_log={config.debug_log}")
    print()


def summarize_available_results() -> None:
    rows = []
    summary_root = REPO_ROOT / "ablation_results" / "p12"
    for mode in ABLATION_SETTINGS:
        run_name = f"gpt5_mini_p12_ablation_{mode}"
        output_dir = mode_output_dir(mode)
        result_csv = output_dir / f"{run_name}_P12_Results.csv"
        detail_csv = output_dir / f"{run_name}_P12_DetailLog.csv"
        if not result_csv.exists():
            continue

        results = pd.read_csv(result_csv)
        global_metrics = compute_global_scaled_metrics(detail_csv)
        if global_metrics is None:
            global_mae = float(results["MAE_scaled"].mean())
            global_mse = float(results["MSE_scaled"].mean())
        else:
            global_mae, global_mse = global_metrics

        rows.append(
            {
                "Ablation_mode": mode,
                "MAE_scaled_global": round(float(global_mae), 6),
                "MSE_scaled_global": round(float(global_mse), 6),
                "MAE_scaled_equal_variable": round(float(results["MAE_scaled"].mean()), 6),
                "MSE_scaled_equal_variable": round(float(results["MSE_scaled"].mean()), 6),
                "Predictions": int(results["Predictions"].sum()),
                "Fallback": int(results["Fallback"].sum()),
                "Result_csv": str(result_csv.relative_to(REPO_ROOT)),
                "Detail_log_csv": str(detail_csv.relative_to(REPO_ROOT)),
            }
        )

    if not rows:
        return

    summary_root.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_root / "ablation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")


def main() -> None:
    args = parse_args()
    configure_common_environment()

    modes = list(ABLATION_SETTINGS) if args.all else [args.mode]
    for mode in modes:
        configure_mode(mode, args.max_test_samples)
        config = ExperimentConfig()
        if args.dry_run:
            print_config(config)
            continue

        require_api_key_if_needed(mode)
        run(config)

    if not args.dry_run:
        summarize_available_results()


if __name__ == "__main__":
    main()
