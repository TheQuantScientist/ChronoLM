"""Run Gemma on the APN PhysioNet 2012 forecasting task.

This experiment uses the APN data pipeline and metrics, but replaces the
trained APN model with zero-shot calls to an OpenAI-compatible LLM endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from chronolm.apn import add_apn_to_path

add_apn_to_path()
from data.dependencies.tsdm.tasks.P12 import Physionet2012  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    api_url: str = os.getenv(
        "CHRONOLM_API_URL",
        "https://they-intranet-medicine-stuff.trycloudflare.com/v1/chat/completions",
    )
    model_id: str = os.getenv("CHRONOLM_MODEL_ID", "google/gemma-3-4b-it")
    api_key: str = os.getenv("CHRONOLM_API_KEY", "litellm-sml-2026")
    seq_len: int = int(os.getenv("CHRONOLM_SEQ_LEN", "36"))
    pred_len: int = int(os.getenv("CHRONOLM_PRED_LEN", "3"))
    max_test_samples: int | None = None
    max_retries: int = 3
    trace_every: int = int(os.getenv("CHRONOLM_TRACE_EVERY", "1"))
    progress_every: int = int(os.getenv("CHRONOLM_PROGRESS_EVERY", "25"))
    trace_history_points: int = int(os.getenv("CHRONOLM_TRACE_HISTORY_POINTS", "5"))
    trace_output_chars: int = int(os.getenv("CHRONOLM_TRACE_OUTPUT_CHARS", "160"))
    temperature: float = 0.6
    max_tokens: int = 100
    request_timeout_s: int = 120
    output_csv: Path = Path("Gemma_temp06_P12_Results.csv")
    checkpoint_csv: Path = Path("Gemma_temp06_P12_Checkpoint.csv")
    detail_log_csv: Path = Path("Gemma_temp06_P12_DetailLog.csv")
    debug_log: Path = Path("gemma_temp06_debug.log")


def configure_logging(log_path: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def hours_to_timestamp(hour_value: float) -> str:
    hours = int(hour_value)
    minutes = int((hour_value - hours) * 60)
    seconds = int(((hour_value - hours) * 60 - minutes) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_numbers(text: str, count: int) -> list[float] | None:
    numbers = re.findall(r"[-+]?\d+\.?\d*", text or "")
    parsed: list[float] = []
    for number in numbers:
        try:
            value = float(number)
        except ValueError:
            continue

        if -10000 < value < 10000:
            parsed.append(value)

    return parsed[:count] if len(parsed) >= count else None


def clamp_to_recent_history(values: Iterable[float], history: list[float]) -> list[float]:
    if len(history) < 3:
        return list(values)

    recent = history[-10:]
    mean = float(np.mean(recent))
    std = max(float(np.std(recent)), 0.1)
    lower = mean - 3 * std
    upper = mean + 3 * std
    return [max(lower, min(upper, value)) for value in values]


def should_trace_sample(valid_sample_count: int, config: ExperimentConfig) -> bool:
    return config.trace_every > 0 and valid_sample_count % config.trace_every == 0


def format_history_tail(
    history_times: list[float],
    history_values: list[float],
    n_points: int,
) -> str:
    tail = list(zip(history_times, history_values))[-n_points:]
    return ", ".join(f"{time_value:.2f}h={value:.2f}" for time_value, value in tail)


def format_values(values: Iterable[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def call_llm(
    config: ExperimentConfig,
    user_prompt: str,
    system_prompt: str,
    logger: logging.Logger,
) -> str:
    full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    try:
        response = requests.post(
            config.api_url,
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.request_timeout_s,
        )
    except Exception as exc:
        logger.info("    [EXCEPTION] %s", str(exc)[:200])
        return ""

    if response.status_code != 200:
        logger.info("    [HTTP_%s] %s", response.status_code, response.text[:200])
        return ""

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        logger.info(
            "    [EMPTY] finish_reason=%s, full=%s",
            data["choices"][0].get("finish_reason"),
            str(data)[:300],
        )
    return content


def load_checkpoint(path: Path, logger: logging.Logger) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []

    try:
        checkpoint = pd.read_csv(path)
    except Exception:
        logger.info("  Checkpoint could not be read. Starting from scratch.")
        return [], []

    records = checkpoint.to_dict("records")
    processed_variables = checkpoint["Variable"].tolist()
    logger.info("  Checkpoint found: %s variables already completed.", len(records))
    logger.info("  %s", processed_variables)
    return records, processed_variables


def build_prompt(variable_name: str, n_forecast: int, history_json: str) -> tuple[str, str]:
    system_prompt = (
        f"Predict next {n_forecast} values for {variable_name}.\n"
        f"Output ONLY {n_forecast} numbers, comma separated. No text.\n\n"
        f"{history_json}"
    )
    return system_prompt, "Predict the next values."


def run(config: ExperimentConfig) -> None:
    warnings.filterwarnings("ignore")
    logger = configure_logging(config.debug_log)

    logger.info("Loading PhysioNet P12 data through the APN pipeline...")
    load_start = time.time()
    task = Physionet2012(seq_len=config.seq_len, pred_len=config.pred_len)
    encoder = task.encoder.column_encoders
    columns = list(task.dataset.columns)
    test_dataset = task.get_dataset((0, "test"))

    total_test_samples = len(test_dataset)
    n_samples = config.max_test_samples or total_test_samples

    logger.info("  Data loaded in %.1fs", time.time() - load_start)
    logger.info("  Test samples: %s, using: %s", total_test_samples, n_samples)
    logger.info("  Variables: %s", len(columns))

    def scaled_to_raw(scaled_value: float, variable_index: int) -> float:
        return float(scaled_value * encoder.stdv[variable_index] + encoder.mean[variable_index])

    def raw_to_scaled(raw_value: float, variable_index: int) -> float:
        return float((raw_value - encoder.mean[variable_index]) / encoder.stdv[variable_index])

    all_results, processed_variables = load_checkpoint(config.checkpoint_csv, logger)

    logger.info("\n%s", "=" * 70)
    logger.info("  %s vs APN - PhysioNet P12", config.model_id)
    logger.info(
        "  Lookback=%sh, forecast=%s steps, one-shot",
        config.seq_len,
        config.pred_len,
    )
    logger.info("  Test samples: %s", n_samples)
    logger.info("%s", "=" * 70)

    total_start = time.time()

    for variable_index, variable_name in enumerate(columns):
        if variable_name in processed_variables:
            logger.info("\n  SKIP: %s already exists in checkpoint", variable_name)
            continue

        logger.info("\n%s", "-" * 70)
        logger.info("  Variable: %s (index=%s)", variable_name, variable_index)
        logger.info("%s", "-" * 70)

        variable_start = time.time()
        mae_scaled: list[float] = []
        mse_scaled: list[float] = []
        mae_raw: list[float] = []
        total_predictions = 0
        fallback_count = 0
        detail_rows: list[dict] = []
        valid_sample_count = 0

        for sample_index in range(n_samples):
            sample = test_dataset[sample_index]
            t_input, x_input, _ = sample.inputs
            y_target = sample.targets

            history_scaled = x_input[:, variable_index]
            target_scaled = y_target[:, variable_index]
            valid_history_mask = ~history_scaled.isnan()
            valid_target_mask = ~target_scaled.isnan()

            if valid_history_mask.sum() < 3 or valid_target_mask.sum() == 0:
                continue

            valid_history = history_scaled[valid_history_mask]
            valid_history_t = t_input[valid_history_mask] * 48
            history_raw = [
                scaled_to_raw(value.item(), variable_index) for value in valid_history
            ]
            history_times = [round(value.item(), 2) for value in valid_history_t]

            series_data = [
                {
                    "timestamp": hours_to_timestamp(timestamp),
                    "value": round(value, 1),
                }
                for timestamp, value in zip(history_times, history_raw)
            ]
            history_json = json.dumps(
                {
                    "symbol": variable_name,
                    "interval": "irregular",
                    "series": series_data,
                },
                indent=2,
            )

            actual_scaled: list[float] = []
            actual_raw: list[float] = []
            for step_index in range(min(config.pred_len, len(target_scaled))):
                if valid_target_mask[step_index]:
                    target_value = target_scaled[step_index].item()
                    actual_scaled.append(target_value)
                    actual_raw.append(scaled_to_raw(target_value, variable_index))

            n_forecast = len(actual_scaled)
            if n_forecast == 0:
                continue

            valid_sample_count += 1
            trace_sample = should_trace_sample(valid_sample_count, config)
            system_prompt, user_prompt = build_prompt(
                variable_name,
                n_forecast,
                history_json,
            )

            predictions_raw: list[float] | None = None
            raw_output = ""
            request_start = time.time()
            if trace_sample:
                logger.info(
                    "    [INPUT] sample=%s/%s valid=%s patient=%s variable=%s "
                    "history_n=%s target_n=%s history_tail=%s",
                    sample_index + 1,
                    n_samples,
                    valid_sample_count,
                    sample.key,
                    variable_name,
                    len(history_raw),
                    n_forecast,
                    format_history_tail(
                        history_times,
                        history_raw,
                        config.trace_history_points,
                    ),
                )

            for attempt in range(config.max_retries):
                if trace_sample:
                    logger.info(
                        "    [CALL] sample=%s patient=%s attempt=%s/%s "
                        "forecast_n=%s prompt_chars=%s",
                        sample_index + 1,
                        sample.key,
                        attempt + 1,
                        config.max_retries,
                        n_forecast,
                        len(system_prompt),
                    )

                raw_output = call_llm(config, user_prompt, system_prompt, logger)
                if not raw_output:
                    if trace_sample:
                        logger.info(
                            "    [OUTPUT] sample=%s patient=%s attempt=%s empty response",
                            sample_index + 1,
                            sample.key,
                            attempt + 1,
                        )
                    time.sleep(2)
                    continue

                if trace_sample:
                    logger.info(
                        "    [OUTPUT] sample=%s patient=%s attempt=%s raw=%r",
                        sample_index + 1,
                        sample.key,
                        attempt + 1,
                        raw_output[: config.trace_output_chars],
                    )

                predictions_raw = parse_numbers(raw_output, n_forecast)
                if predictions_raw is not None:
                    if trace_sample:
                        logger.info(
                            "    [PARSE] sample=%s patient=%s parsed_raw=%s",
                            sample_index + 1,
                            sample.key,
                            format_values(predictions_raw),
                        )
                    break
                if trace_sample:
                    logger.info(
                        "    [PARSE] sample=%s patient=%s failed to parse enough numbers",
                        sample_index + 1,
                        sample.key,
                    )

            if predictions_raw is None:
                predictions_raw = [history_raw[-1]] * n_forecast
                fallback_count += 1
                if trace_sample or fallback_count <= 5:
                    logger.info(
                        "    [FALLBACK] Patient=%s, output=%r",
                        sample.key,
                        raw_output[:100],
                    )

            unclamped_predictions_raw = list(predictions_raw)
            predictions_raw = clamp_to_recent_history(predictions_raw, history_raw)
            if trace_sample and predictions_raw != unclamped_predictions_raw:
                logger.info(
                    "    [CLAMP] sample=%s patient=%s raw=%s clamped=%s",
                    sample_index + 1,
                    sample.key,
                    format_values(unclamped_predictions_raw),
                    format_values(predictions_raw),
                )

            sample_scaled_errors = []
            sample_raw_errors = []
            for step_index in range(n_forecast):
                predicted_scaled = raw_to_scaled(
                    predictions_raw[step_index],
                    variable_index,
                )
                scaled_error = abs(actual_scaled[step_index] - predicted_scaled)

                mae_scaled.append(scaled_error)
                mse_scaled.append((actual_scaled[step_index] - predicted_scaled) ** 2)
                raw_error = abs(actual_raw[step_index] - predictions_raw[step_index])
                mae_raw.append(raw_error)
                sample_scaled_errors.append(scaled_error)
                sample_raw_errors.append(raw_error)

                detail_rows.append(
                    {
                        "Variable": variable_name,
                        "Patient": sample.key,
                        "Step": step_index,
                        "Actual_scaled": round(actual_scaled[step_index], 6),
                        "Predicted_scaled": round(predicted_scaled, 6),
                        "Actual_raw": round(actual_raw[step_index], 4),
                        "Predicted_raw": round(predictions_raw[step_index], 4),
                        "AE_scaled": round(scaled_error, 6),
                    }
                )
                total_predictions += 1

            if trace_sample:
                logger.info(
                    "    [SCORE] sample=%s patient=%s actual_raw=%s predicted_raw=%s "
                    "mae_scaled=%.4f mae_raw=%.4f latency=%.1fs",
                    sample_index + 1,
                    sample.key,
                    format_values(actual_raw),
                    format_values(predictions_raw),
                    float(np.mean(sample_scaled_errors)),
                    float(np.mean(sample_raw_errors)),
                    time.time() - request_start,
                )

            if (
                (sample_index + 1) % config.progress_every == 0
                or (sample_index + 1) == n_samples
            ):
                elapsed = time.time() - variable_start
                speed = (sample_index + 1) / elapsed if elapsed > 0 else 0
                remaining = (n_samples - sample_index - 1) / speed if speed > 0 else 0
                logger.info(
                    "  [%s] scanned=%s/%s valid=%s predictions=%s fallback=%s "
                    "(%.0fs elapsed, ~%.0fs remaining)",
                    variable_name,
                    sample_index + 1,
                    n_samples,
                    valid_sample_count,
                    total_predictions,
                    fallback_count,
                    elapsed,
                    remaining,
                )

        variable_elapsed = time.time() - variable_start

        if not mae_scaled:
            logger.info("  %s: no valid target data, skipped", variable_name)
            continue

        result_row = {
            "Variable": variable_name,
            "Model": config.model_id,
            "MAE_scaled": round(float(np.mean(mae_scaled)), 6),
            "MSE_scaled": round(float(np.mean(mse_scaled)), 6),
            "MAE_raw": round(float(np.mean(mae_raw)), 4),
            "Predictions": total_predictions,
            "Fallback": fallback_count,
            "Time_s": round(variable_elapsed, 1),
        }
        all_results.append(result_row)

        logger.info(
            "\n  %s: MAE_scaled=%s, MSE_scaled=%s, MAE_raw=%s, "
            "fallback=%s, time=%ss",
            variable_name,
            result_row["MAE_scaled"],
            result_row["MSE_scaled"],
            result_row["MAE_raw"],
            fallback_count,
            result_row["Time_s"],
        )

        pd.DataFrame(all_results).to_csv(config.checkpoint_csv, index=False)
        logger.info("  Checkpoint saved (%s/%s variables)", len(all_results), len(columns))

        if detail_rows:
            pd.DataFrame(detail_rows).to_csv(
                config.detail_log_csv,
                mode="a",
                header=not config.detail_log_csv.exists(),
                index=False,
            )

    if not all_results:
        logger.info("No results were produced.")
        return

    results = pd.DataFrame(all_results)
    results.to_csv(config.output_csv, index=False)

    total_elapsed = time.time() - total_start
    avg_mae = results["MAE_scaled"].mean()
    avg_mse = results["MSE_scaled"].mean()
    total_predictions = results["Predictions"].sum()
    total_fallback = results["Fallback"].sum()

    logger.info("\n%s", "=" * 70)
    logger.info("  %s vs APN - PhysioNet P12 results", config.model_id)
    logger.info("%s", "=" * 70)
    logger.info(
        "\n%s",
        results[
            [
                "Variable",
                "MAE_scaled",
                "MSE_scaled",
                "MAE_raw",
                "Predictions",
                "Fallback",
            ]
        ].to_string(index=False),
    )
    logger.info("\n%s", "-" * 70)
    logger.info("  Average across variables:")
    logger.info("    MAE_scaled = %.6f", avg_mae)
    logger.info("    MSE_scaled = %.6f", avg_mse)
    logger.info("    Total predictions: %s", f"{total_predictions:,}")
    logger.info("    Total fallback: %s", total_fallback)
    logger.info("    Total time: %.0fs (%.1fh)", total_elapsed, total_elapsed / 3600)
    logger.info("  APN paper baseline: MAE=0.3762, MSE=0.2936")
    logger.info("  Gemma result:       MAE=%.4f, MSE=%.4f", avg_mae, avg_mse)

    if avg_mae < 0.3762:
        logger.info("  Gemma beats APN by %.4f MAE.", 0.3762 - avg_mae)
    else:
        logger.info("  APN remains better by %.4f MAE.", avg_mae - 0.3762)

    logger.info("%s", "=" * 70)
    logger.info("Saved results: %s", config.output_csv)
    logger.info("Saved detail log: %s", config.detail_log_csv)
    logger.info("Saved debug log: %s", config.debug_log)
    logger.info("Saved checkpoint: %s", config.checkpoint_csv)

    if len(all_results) >= len(columns):
        logger.info("All %s variables are complete.", len(columns))


def main() -> None:
    run(ExperimentConfig())


if __name__ == "__main__":
    main()
