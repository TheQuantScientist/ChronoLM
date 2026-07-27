"""Run an LLM on supported APN forecasting benchmark tasks.

This experiment uses APN data pipelines and metrics, but replaces the trained
APN model with zero-shot calls to an OpenAI-compatible LLM endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests

from chronolm.apn import APN_ROOT, add_apn_to_path

add_apn_to_path()


DATASET_DEFAULTS = {
    "P12": {"seq_len": 36, "pred_len": 3, "display_name": "PhysioNet P12"},
    "USHCN": {"seq_len": 150, "pred_len": 3, "display_name": "USHCN"},
    "HumanActivity": {
        "seq_len": 3000,
        "pred_len": 300,
        "display_name": "HumanActivity",
    },
}

ANCHOR_BLEND_PROMPT_STYLES = {
    "gpt_climate_anchor_blend_mse",
    "gpt_clinical_anchor_blend_mse",
    "gpt_activity_anchor_blend_mse",
}

APN_BASELINES = {
    "HumanActivity": {"MAE": 0.1159, "MSE": 0.0421},
    "P12": {"MAE": 0.3762, "MSE": 0.2936},
    "USHCN": {"MAE": 0.2611, "MSE": 0.1590},
}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off")


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def env_optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def slugify_model_name(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").lower()
    return slug or "llm"


def canonical_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().replace("-", "_").upper()
    aliases = {
        "PHYSIONET": "P12",
        "PHYSIONET2012": "P12",
        "PHYSIONET_2012": "P12",
        "P12": "P12",
        "USHCN": "USHCN",
        "USHCN_DEBROUWER2019": "USHCN",
        "HUMANACTIVITY": "HumanActivity",
        "HUMAN_ACTIVITY": "HumanActivity",
        "PERSONACTIVITY": "HumanActivity",
        "PERSON_ACTIVITY": "HumanActivity",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_DEFAULTS))
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Use one of: {supported}") from exc


@dataclass(frozen=True)
class ExperimentConfig:
    api_url: str = field(
        default_factory=lambda: os.getenv(
            "CHRONOLM_API_URL",
            "https://they-intranet-medicine-stuff.trycloudflare.com/v1/chat/completions",
        )
    )
    api_provider: str = field(
        default_factory=lambda: os.getenv("CHRONOLM_API_PROVIDER", "openai-compatible")
    )
    model_id: str = field(default_factory=lambda: os.getenv("CHRONOLM_MODEL_ID", "Qwen/Qwen3.5-4B"))
    api_key: str = field(default_factory=lambda: os.getenv("CHRONOLM_API_KEY", "litellm-sml-2026"))
    run_name: str | None = field(default_factory=lambda: os.getenv("CHRONOLM_RUN_NAME"))
    dataset_name: str = field(default_factory=lambda: os.getenv("CHRONOLM_DATASET", "P12"))
    prompt_style: str = field(default_factory=lambda: os.getenv("CHRONOLM_PROMPT_STYLE", "compact"))
    seq_len: int = field(default_factory=lambda: env_int("CHRONOLM_SEQ_LEN", 36))
    pred_len: int = field(default_factory=lambda: env_int("CHRONOLM_PRED_LEN", 3))
    max_test_samples: int | None = field(default_factory=lambda: env_optional_int("CHRONOLM_MAX_TEST_SAMPLES"))
    max_retries: int = field(default_factory=lambda: env_int("CHRONOLM_MAX_RETRIES", 6))
    trace_every: int = field(default_factory=lambda: env_int("CHRONOLM_TRACE_EVERY", 1))
    progress_every: int = field(default_factory=lambda: env_int("CHRONOLM_PROGRESS_EVERY", 25))
    trace_history_points: int = field(default_factory=lambda: env_int("CHRONOLM_TRACE_HISTORY_POINTS", 5))
    trace_output_chars: int = field(default_factory=lambda: env_int("CHRONOLM_TRACE_OUTPUT_CHARS", 160))
    temperature: float = field(default_factory=lambda: env_float("CHRONOLM_TEMPERATURE", 0.6))
    include_temperature: bool = field(default_factory=lambda: env_bool("CHRONOLM_INCLUDE_TEMPERATURE", True))
    enable_thinking: bool = field(default_factory=lambda: env_bool("CHRONOLM_ENABLE_THINKING", False))
    reasoning_effort: str | None = field(default_factory=lambda: os.getenv("CHRONOLM_REASONING_EFFORT"))
    verbosity: str | None = field(default_factory=lambda: os.getenv("CHRONOLM_VERBOSITY"))
    max_tokens: int = field(default_factory=lambda: env_int("CHRONOLM_MAX_TOKENS", 100))
    anchor_blend_weight: float = field(
        default_factory=lambda: env_float("CHRONOLM_ANCHOR_BLEND_WEIGHT", 0.9)
    )
    request_timeout_s: int = field(default_factory=lambda: env_int("CHRONOLM_REQUEST_TIMEOUT_S", 120))
    output_csv: Path | None = field(default_factory=lambda: env_optional_path("CHRONOLM_OUTPUT_CSV"))
    checkpoint_csv: Path | None = field(default_factory=lambda: env_optional_path("CHRONOLM_CHECKPOINT_CSV"))
    detail_log_csv: Path | None = field(default_factory=lambda: env_optional_path("CHRONOLM_DETAIL_LOG_CSV"))
    debug_log: Path | None = field(default_factory=lambda: env_optional_path("CHRONOLM_DEBUG_LOG"))

    def __post_init__(self) -> None:
        dataset_name = canonical_dataset_name(self.dataset_name)
        defaults = DATASET_DEFAULTS[dataset_name]
        object.__setattr__(self, "dataset_name", dataset_name)
        if "CHRONOLM_SEQ_LEN" not in os.environ:
            object.__setattr__(self, "seq_len", defaults["seq_len"])
        if "CHRONOLM_PRED_LEN" not in os.environ:
            object.__setattr__(self, "pred_len", defaults["pred_len"])

        run_name = self.run_name or slugify_model_name(self.model_id)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(
            self,
            "output_csv",
            self.output_csv or Path(f"{run_name}_{dataset_name}_Results.csv"),
        )
        object.__setattr__(
            self,
            "checkpoint_csv",
            self.checkpoint_csv or Path(f"{run_name}_{dataset_name}_Checkpoint.csv"),
        )
        object.__setattr__(
            self,
            "detail_log_csv",
            self.detail_log_csv or Path(f"{run_name}_{dataset_name}_DetailLog.csv"),
        )
        object.__setattr__(
            self,
            "debug_log",
            self.debug_log or Path(f"{run_name}_{dataset_name}_debug.log"),
        )


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


def strip_thinking_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)


def parse_numbers(text: str, count: int) -> list[float] | None:
    cleaned = strip_thinking_content(text).strip()
    if re.search(r"[A-Za-z]", cleaned):
        return None

    number_pattern = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"
    numbers = re.findall(number_pattern, cleaned)
    separators = re.sub(number_pattern, "", cleaned)
    if re.search(r"[^,\s;\[\]\(\).]", separators):
        return None

    if len(numbers) != count:
        return None

    parsed: list[float] = []
    for number in numbers:
        try:
            value = float(number)
        except ValueError:
            continue

        if -10000 < value < 10000:
            parsed.append(value)

    return parsed if len(parsed) == count else None


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
    time_formatter: Callable[[float], str],
) -> str:
    tail = list(zip(history_times, history_values))[-n_points:]
    return ", ".join(f"{time_formatter(time_value)}={value:.2f}" for time_value, value in tail)


def format_values(values: Iterable[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def round_prompt_value(value: float, dataset_name: str) -> float:
    if dataset_name == "USHCN":
        return round(float(value), 4)
    if dataset_name == "HumanActivity":
        return round(float(value), 3)
    return round(float(value), 1)


def most_common_rounded(values: list[float], digits: int = 4) -> float:
    rounded_values = [round(float(value), digits) for value in values]
    return float(Counter(rounded_values).most_common(1)[0][0])


def ushcn_annual_phase_anchor(
    history_times: list[float],
    history_values: list[float],
    target_time: float,
    period: float = 50.0,
) -> float:
    times = np.asarray(history_times, dtype=float)
    values = np.asarray(history_values, dtype=float)
    phase_distance = np.abs(((times - target_time + period / 2) % period) - period / 2)
    nearest = np.argsort(phase_distance)[: min(5, len(values))]
    weights = 1.0 / (phase_distance[nearest] + 0.25)
    return float(np.average(values[nearest], weights=weights))


def ushcn_recent_trend_anchor(
    history_times: list[float],
    history_values: list[float],
    target_time: float,
) -> float:
    k = min(8, len(history_values))
    times = np.asarray(history_times[-k:], dtype=float)
    values = np.asarray(history_values[-k:], dtype=float)

    if len(values) < 3 or float(np.ptp(times)) <= 1e-9:
        return float(history_values[-1])

    slope, intercept = np.polyfit(times, values, 1)
    trend = float(intercept + slope * target_time)
    low = float(np.min(values))
    high = float(np.max(values))
    padding = max(0.25, 0.25 * (high - low))
    return float(np.clip(trend, low - padding, high + padding))


def build_ushcn_forecast_anchors(
    variable_name: str,
    history_times: list[float],
    history_values: list[float],
    target_times: list[float],
) -> list[dict[str, float | int | str]]:
    mode_value = most_common_rounded(history_values)
    last_value = float(history_values[-1])

    anchors: list[dict[str, float | int | str]] = []
    for step_index, target_time in enumerate(target_times, start=1):
        phase_value = ushcn_annual_phase_anchor(history_times, history_values, target_time)
        trend_value = ushcn_recent_trend_anchor(history_times, history_values, target_time)

        if variable_name in {"Value_0", "Value_1"}:
            recommended = mode_value
            rule = "sparse_mode"
        elif variable_name == "Value_2":
            recommended = 0.8 * last_value + 0.2 * phase_value
            rule = "persistence_with_phase"
        elif variable_name == "Value_3":
            recommended = 0.75 * phase_value + 0.25 * last_value
            rule = "annual_phase_dominant"
        elif variable_name == "Value_4":
            recommended = 0.65 * phase_value + 0.24 * trend_value + 0.11 * last_value
            rule = "annual_phase_plus_trend"
        else:
            recommended = 0.5 * phase_value + 0.5 * last_value
            rule = "phase_persistence_blend"

        anchors.append(
            {
                "step": step_index,
                "target_time": round(float(target_time), 2),
                "recommended_anchor": round(float(recommended), 4),
                "rule": rule,
            }
        )

    return anchors


def build_p12_forecast_anchors(
    history_values: list[float],
    target_times: list[float],
) -> list[dict[str, float | int | str]]:
    recent_values = history_values[-3:]
    recent_mean = float(np.mean(recent_values))

    return [
        {
            "step": step_index,
            "target_time": round(float(target_time), 2),
            "recommended_anchor": round(recent_mean, 4),
            "rule": "recent_mean3",
        }
        for step_index, target_time in enumerate(target_times, start=1)
    ]


def build_activity_forecast_anchors(
    history_times: list[float],
    history_values: list[float],
    target_times: list[float],
) -> list[dict[str, float | int | str]]:
    recent_count = min(12, len(history_values))
    recent_times = np.asarray(history_times[-recent_count:], dtype=float)
    recent_values = np.asarray(history_values[-recent_count:], dtype=float)

    last_value = float(recent_values[-1])
    recent_mean = float(np.mean(recent_values))
    recent_std = float(np.std(recent_values))
    recent_low = float(np.min(recent_values))
    recent_high = float(np.max(recent_values))
    time_span = float(np.ptp(recent_times))

    slope = 0.0
    intercept = last_value
    if len(recent_values) >= 3 and time_span > 1e-9:
        weights = np.linspace(0.55, 1.0, len(recent_values))
        slope, intercept = np.polyfit(recent_times, recent_values, 1, w=weights)

    diffs = np.diff(recent_values)
    directional_consistency = 0.0
    if len(diffs) > 0 and np.any(np.abs(diffs) > 1e-9):
        directional_consistency = abs(float(np.mean(np.sign(diffs[np.abs(diffs) > 1e-9]))))

    anchors: list[dict[str, float | int | str]] = []
    for step_index, target_time in enumerate(target_times, start=1):
        if recent_std < 0.05 or time_span <= 1e-9:
            recommended = 0.9 * last_value + 0.1 * recent_mean
            rule = "stable_persistence"
        else:
            trend_value = float(intercept + slope * target_time)
            gap_ms = max(0.0, float(target_time) - float(recent_times[-1]))
            horizon_fraction = min(1.0, gap_ms / 300.0)
            trend_weight = 0.45 + 0.25 * directional_consistency - 0.15 * horizon_fraction
            trend_weight = float(np.clip(trend_weight, 0.25, 0.70))
            persistent_level = 0.8 * last_value + 0.2 * recent_mean
            recommended = trend_weight * trend_value + (1.0 - trend_weight) * persistent_level
            padding = max(0.08, 1.25 * recent_std)
            recommended = float(
                np.clip(recommended, recent_low - padding, recent_high + padding)
            )
            rule = "damped_velocity"

        anchors.append(
            {
                "step": step_index,
                "target_time": round(float(target_time), 1),
                "recommended_anchor": round(float(recommended), 4),
                "rule": rule,
            }
        )

    return anchors


def build_recent_multichannel_tail(
    t_input: object,
    x_input: object,
    columns: list[str],
    benchmark: "BenchmarkData",
    max_rows: int = 12,
) -> list[dict[str, object]]:
    rows = []
    start_index = max(0, len(t_input) - max_rows)
    for row_index in range(start_index, len(t_input)):
        values = {}
        for column_index, column_name in enumerate(columns):
            value = x_input[row_index, column_index]
            if not bool(value.isnan().item()):
                values[column_name] = round(float(value.item()), 4)

        if values:
            timestamp = benchmark.time_to_prompt(float(t_input[row_index].item()))
            rows.append(
                {
                    "timestamp": benchmark.format_timestamp(timestamp),
                    "values": values,
                }
            )

    return rows


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    display_name: str
    test_dataset: object
    columns: list[str]
    encoder_index_by_column: dict[str, int]
    scaled_to_raw: Callable[[float, int], float]
    raw_to_scaled: Callable[[float, int], float]
    time_to_prompt: Callable[[float], float]
    format_timestamp: Callable[[float], str]
    format_tail_time: Callable[[float], str]
    time_unit: str
    entity_label: str


@dataclass(frozen=True)
class ForecastSample:
    key: str | int
    inputs: tuple[object, object, object]
    targets: object


def build_human_activity_samples(
    records: Iterable[tuple[object, object, object, object]],
    seq_len: int,
    pred_len: int,
) -> list[ForecastSample]:
    import torch

    samples: list[ForecastSample] = []
    for record_id, tt, vals, mask in records:
        t_max = int(tt.max())
        for start_time in range(0, t_max - seq_len, 4000):
            end_history = start_time + seq_len
            end_target = start_time + seq_len + pred_len

            if end_history >= t_max:
                history_indices = torch.where((tt >= start_time) & (tt <= end_history))[0]
            else:
                history_indices = torch.where((tt >= start_time) & (tt < end_history))[0]

            if end_target >= t_max:
                target_indices = torch.where((tt >= end_history) & (tt <= end_target))[0]
            else:
                target_indices = torch.where((tt >= end_history) & (tt < end_target))[0]

            if len(history_indices) == 0:
                continue

            x_values = vals[history_indices].clone().float()
            x_mask = mask[history_indices].bool()
            y_values = vals[target_indices].clone().float()
            y_mask = mask[target_indices].bool()

            x_values[~x_mask] = torch.nan
            y_values[~y_mask] = torch.nan

            t_input = (tt[history_indices] - start_time).float()
            t_target = (tt[target_indices] - start_time).float()
            sample_key = f"{record_id}_{start_time // pred_len}"

            samples.append(
                ForecastSample(
                    key=sample_key,
                    inputs=(t_input, x_values, t_target),
                    targets=y_values,
                )
            )

    return samples


def load_benchmark_data(config: ExperimentConfig) -> BenchmarkData:
    if config.dataset_name == "P12":
        from data.dependencies.tsdm.tasks.P12 import Physionet2012  # noqa: E402

        task = Physionet2012(seq_len=config.seq_len, pred_len=config.pred_len)
        encoder = task.encoder.column_encoders
        columns = list(task.dataset.columns)
        encoder_columns = list(task.encoder.columns)
        encoder_index_by_column = {
            column_name: column_index
            for column_index, column_name in enumerate(encoder_columns)
        }

        def scaled_to_raw(scaled_value: float, encoder_index: int) -> float:
            return float(scaled_value * encoder.stdv[encoder_index] + encoder.mean[encoder_index])

        def raw_to_scaled(raw_value: float, encoder_index: int) -> float:
            return float((raw_value - encoder.mean[encoder_index]) / encoder.stdv[encoder_index])

        return BenchmarkData(
            name="P12",
            display_name="PhysioNet P12",
            test_dataset=task.get_dataset((0, "test")),
            columns=columns,
            encoder_index_by_column=encoder_index_by_column,
            scaled_to_raw=scaled_to_raw,
            raw_to_scaled=raw_to_scaled,
            time_to_prompt=lambda value: value * 48,
            format_timestamp=hours_to_timestamp,
            format_tail_time=lambda value: f"{value:.2f}h",
            time_unit="hours after ICU admission",
            entity_label="patient",
        )

    if config.dataset_name == "USHCN":
        from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import (  # noqa: E402
            USHCN_DeBrouwer2019,
        )

        task = USHCN_DeBrouwer2019(
            normalize_time=False,
            seq_len=config.seq_len - 0.5,
            pred_len=config.pred_len,
        )
        columns = list(task.dataset.columns)

        return BenchmarkData(
            name="USHCN",
            display_name="USHCN",
            test_dataset=task.get_dataset((0, "test")),
            columns=columns,
            encoder_index_by_column={
                column_name: column_index
                for column_index, column_name in enumerate(columns)
            },
            scaled_to_raw=lambda value, _index: float(value),
            raw_to_scaled=lambda value, _index: float(value),
            time_to_prompt=lambda value: value,
            format_timestamp=lambda value: f"day {value:.2f}",
            format_tail_time=lambda value: f"{value:.2f}d",
            time_unit="days in the 4-year USHCN window",
            entity_label="station",
        )

    if config.dataset_name == "HumanActivity":
        from data.dependencies.HumanActivity.HumanActivity import HumanActivity  # noqa: E402
        from sklearn import model_selection  # noqa: E402

        dataset_root = Path(
            os.getenv(
                "CHRONOLM_HUMANACTIVITY_ROOT",
                str(APN_ROOT / "storage" / "datasets" / "HumanActivity"),
            )
        )
        processed_path = dataset_root / "processed" / "data.pt"
        human_activity = HumanActivity(
            root=str(dataset_root),
            download=not processed_path.exists(),
        )
        _seen_data, test_records = model_selection.train_test_split(
            human_activity,
            train_size=0.9,
            random_state=42,
            shuffle=False,
        )

        tag_names = ["ANKLE_LEFT", "ANKLE_RIGHT", "CHEST", "BELT"]
        axis_names = ["x", "y", "z"]
        columns = [f"{tag}_{axis}" for tag in tag_names for axis in axis_names]

        return BenchmarkData(
            name="HumanActivity",
            display_name="HumanActivity",
            test_dataset=build_human_activity_samples(
                test_records,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
            ),
            columns=columns,
            encoder_index_by_column={
                column_name: column_index
                for column_index, column_name in enumerate(columns)
            },
            scaled_to_raw=lambda value, _index: float(value),
            raw_to_scaled=lambda value, _index: float(value),
            time_to_prompt=lambda value: value,
            format_timestamp=lambda value: f"{value:.0f} ms",
            format_tail_time=lambda value: f"{value:.0f}ms",
            time_unit="milliseconds inside the APN HumanActivity activity window",
            entity_label="activity_window",
        )

    raise ValueError(f"Unsupported dataset: {config.dataset_name}")


def uses_openai_chat_api(config: ExperimentConfig) -> bool:
    return (
        config.api_provider.strip().lower() == "openai"
        or config.api_url.rstrip("/") == "https://api.openai.com/v1/chat/completions"
    )


def call_llm(
    config: ExperimentConfig,
    user_prompt: str,
    system_prompt: str,
    logger: logging.Logger,
) -> str:
    full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    openai_chat_api = uses_openai_chat_api(config)
    if not config.enable_thinking and not openai_chat_api:
        full_prompt = f"/no_think\n{full_prompt}"

    payload = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": full_prompt}],
    }
    if openai_chat_api:
        payload["max_completion_tokens"] = config.max_tokens
        if config.reasoning_effort:
            payload["reasoning_effort"] = config.reasoning_effort
        if config.verbosity:
            payload["verbosity"] = config.verbosity
        if config.include_temperature:
            payload["temperature"] = config.temperature
    else:
        payload["max_tokens"] = config.max_tokens
        payload["temperature"] = config.temperature
        payload["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}

    try:
        response = requests.post(
            config.api_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.request_timeout_s,
        )
    except Exception as exc:
        logger.info("    [EXCEPTION] %s", str(exc)[:200])
        return ""

    if (
        response.status_code == 400
        and "temperature" in payload
        and "temperature" in response.text.lower()
        and "unsupported" in response.text.lower()
    ):
        logger.info("    [TEMPERATURE] server rejected temperature; retrying without it")
        payload.pop("temperature")
        try:
            response = requests.post(
                config.api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=config.request_timeout_s,
            )
        except Exception as exc:
            logger.info("    [EXCEPTION] %s", str(exc)[:200])
            return ""

    if response.status_code == 422 and "chat_template_kwargs" in payload:
        logger.info("    [THINKING] server rejected chat_template_kwargs; retrying with /no_think only")
        payload.pop("chat_template_kwargs")
        try:
            response = requests.post(
                config.api_url,
                json=payload,
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


def build_prompt(
    variable_name: str,
    n_forecast: int,
    history_json: str,
    config: ExperimentConfig,
) -> tuple[str, str]:
    value_label = "number" if n_forecast == 1 else "numbers"
    step_label = "next step" if n_forecast == 1 else f"next {n_forecast} steps"

    if config.prompt_style == "gpt_clinical_mse":
        system_prompt = (
            f"You are a healthcare time-series forecaster for {variable_name}.\n"
            "Use the exact target_timestamps. They are known at inference time; only "
            "the target values are hidden.\n"
            "Silently assess the history for trend, level shifts, irregular time gaps, "
            "and clinically plausible sudden changes.\n"
            "Predict the conditional expected next values, optimized for low squared "
            "error: avoid speculative spikes unless the recent evidence supports them.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, comma "
            "separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the next values."

    if config.prompt_style == "gpt_clinical_anchor_blend_mse":
        system_prompt = (
            f"You are a healthcare time-series forecaster for {variable_name} using "
            "history-derived anchors.\n"
            "Use the exact target_timestamps. They are known at inference time; only "
            "the target values are hidden.\n"
            "The forecast_anchors are computed only from observed history. Treat each "
            "recommended_anchor as a conservative recent-level prior.\n"
            "Silently assess the history for trend, level shifts, irregular time gaps, "
            "and clinically plausible sudden changes.\n"
            "Optimize for low squared error: stay near the anchor when evidence is "
            "weak, and move away only when recent history strongly supports a trend or "
            "clinically plausible regime change.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, in target "
            "timestamp order, comma separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the target-timestamp clinical values."

    if config.prompt_style == "gpt_climate_mse":
        system_prompt = (
            "You are forecasting one anonymized USHCN climate channel from a single "
            "meteorological station.\n"
            f"Channel: {variable_name}. Values are already on the benchmark numeric "
            "scale; do not convert units.\n"
            "Silently infer the local level, persistence, recent slope, irregular time "
            "gaps, and any stable seasonal or regime pattern visible in the history.\n"
            "Predict conditional expected values optimized for low squared error. "
            "Prefer persistence plus recent trend when evidence is weak; avoid sudden "
            "jumps or reversals unless the latest observations strongly support them.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, comma "
            "separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the next climate values."

    if config.prompt_style == "gpt_climate_time_mse":
        system_prompt = (
            "You are a USHCN irregular climate time-series forecaster.\n"
            f"Forecast channel {variable_name} for one station. Values are already on "
            "the benchmark numeric scale; do not convert units.\n"
            "Use the exact target_timestamps. They are known at inference time; only "
            "the target values are hidden.\n"
            "Infer the conditional mean at each target timestamp from the same-channel "
            "history, irregular gaps, local level, persistence, recent slope, annual "
            "phase/seasonal recurrence, and the recent multichannel station context.\n"
            "Optimize for low MSE: reduce large misses. For stable sparse channels, "
            "prefer the dominant/persistent level unless recent observations strongly "
            "support an event. For oscillating climate channels, use the target-time "
            "seasonal phase rather than only the last value.\n"
            "Avoid unsupported spikes, but do not over-smooth clear phase reversals or "
            "recent sustained changes.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, in target "
            "timestamp order, comma separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the target-timestamp climate values."

    if config.prompt_style == "gpt_climate_anchor_blend_mse":
        system_prompt = (
            "You are a USHCN irregular climate time-series forecaster using "
            "history-derived anchors.\n"
            f"Forecast channel {variable_name} for one station. Values are already on "
            "the benchmark numeric scale; do not convert units.\n"
            "Use exact target_timestamps. They are known at inference time; only target "
            "values are hidden.\n"
            "The forecast_anchors are computed only from observed history and target "
            "timestamps. Treat each recommended_anchor as a conservative conditional "
            "mean prior. Improve it only when same-channel history and recent "
            "multichannel context give clear evidence.\n"
            "Optimize MSE: avoid large misses. Stay close to the anchor when uncertain; "
            "move away only for strong trend, phase, or regime evidence.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, in target "
            "timestamp order, comma separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the target-timestamp climate values."

    if config.prompt_style == "gpt_activity_anchor_blend_mse":
        system_prompt = (
            "You are a wearable-sensor HumanActivity forecaster using "
            "history-derived anchors.\n"
            f"Forecast accelerometer channel {variable_name} for one short activity "
            "window. Values are already on the benchmark numeric sensor scale; do "
            "not convert units.\n"
            "Use exact target_timestamps. They are known at inference time; only "
            "target values are hidden.\n"
            "The forecast_anchors are computed only from same-channel observed "
            "history using conservative persistence and damped recent velocity. "
            "Treat each recommended_anchor as a low-MSE prior.\n"
            "Use the same-channel history and recent_multichannel_tail to identify "
            "local motion direction, acceleration/deceleration, sensor synchrony, "
            "and abrupt activity transitions. Optimize MSE: stay close to the anchor "
            "when evidence is weak, preserve clear recent motion when it is "
            "consistent across timestamps, and avoid unsupported spikes.\n"
            f"Output ONLY {n_forecast} {value_label} for the {step_label}, in target "
            "timestamp order, comma separated. No text.\n\n"
            f"{history_json}"
        )
        return system_prompt, "Forecast the target-timestamp activity values."

    system_prompt = (
        f"Predict next {n_forecast} values for {variable_name}.\n"
        f"Output ONLY {n_forecast} numbers, comma separated. No text.\n\n"
        f"{history_json}"
    )
    return system_prompt, "Predict the next values."


def compute_global_scaled_metrics(detail_log_csv: Path) -> tuple[float, float] | None:
    if not detail_log_csv.exists():
        return None

    detail = pd.read_csv(detail_log_csv)
    required = {"Actual_scaled", "Predicted_scaled"}
    if not required <= set(detail.columns):
        return None

    valid = detail[["Actual_scaled", "Predicted_scaled"]].notna().all(axis=1)
    detail = detail.loc[valid]
    if detail.empty:
        return None

    residual = detail["Actual_scaled"] - detail["Predicted_scaled"]
    return float(residual.abs().mean()), float((residual**2).mean())


def run(config: ExperimentConfig) -> None:
    warnings.filterwarnings("ignore")
    logger = configure_logging(config.debug_log)

    logger.info("Run name: %s", config.run_name)
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Model: %s", config.model_id)
    logger.info("Prompt style: %s", config.prompt_style)
    if config.prompt_style in ANCHOR_BLEND_PROMPT_STYLES:
        logger.info("Anchor blend weight: %.2f", config.anchor_blend_weight)
    logger.info("API URL: %s", config.api_url)
    logger.info("Thinking mode: %s", "on" if config.enable_thinking else "off")
    logger.info("Loading %s data through the APN pipeline...", config.dataset_name)
    load_start = time.time()
    benchmark = load_benchmark_data(config)
    columns = benchmark.columns
    test_dataset = benchmark.test_dataset

    total_test_samples = len(test_dataset)
    n_samples = config.max_test_samples or total_test_samples

    logger.info("  Data loaded in %.1fs", time.time() - load_start)
    logger.info("  Test samples: %s, using: %s", total_test_samples, n_samples)
    logger.info("  Variables: %s", len(columns))

    all_results, processed_variables = load_checkpoint(config.checkpoint_csv, logger)

    logger.info("\n%s", "=" * 70)
    logger.info("  %s vs APN - %s", config.model_id, benchmark.display_name)
    logger.info(
        "  Lookback=%s, forecast=%s, one-shot",
        config.seq_len,
        config.pred_len,
    )
    logger.info("  Test samples: %s", n_samples)
    logger.info("%s", "=" * 70)

    total_start = time.time()

    for variable_index, variable_name in enumerate(columns):
        encoder_index = benchmark.encoder_index_by_column[variable_name]
        if variable_name in processed_variables:
            logger.info("\n  SKIP: %s already exists in checkpoint", variable_name)
            continue

        logger.info("\n%s", "-" * 70)
        logger.info(
            "  Variable: %s (dataset_index=%s, encoder_index=%s)",
            variable_name,
            variable_index,
            encoder_index,
        )
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
            t_input, x_input, t_target = sample.inputs
            y_target = sample.targets

            history_scaled = x_input[:, variable_index]
            target_scaled = y_target[:, variable_index]
            valid_history_mask = ~history_scaled.isnan()
            valid_target_mask = ~target_scaled.isnan()

            if valid_history_mask.sum() < 3 or valid_target_mask.sum() == 0:
                continue

            valid_history = history_scaled[valid_history_mask]
            valid_history_t = t_input[valid_history_mask]
            history_raw = [
                benchmark.scaled_to_raw(value.item(), encoder_index) for value in valid_history
            ]
            history_times = [
                round(benchmark.time_to_prompt(value.item()), 2)
                for value in valid_history_t
            ]

            actual_scaled: list[float] = []
            actual_raw: list[float] = []
            target_times: list[float] = []
            target_schedule: list[dict[str, object]] = []
            forecast_limit = min(config.pred_len, len(target_scaled), len(t_target))
            for step_index in range(forecast_limit):
                if valid_target_mask[step_index]:
                    target_value = target_scaled[step_index].item()
                    target_time = round(
                        benchmark.time_to_prompt(float(t_target[step_index].item())),
                        2,
                    )
                    actual_scaled.append(target_value)
                    actual_raw.append(benchmark.scaled_to_raw(target_value, encoder_index))
                    target_times.append(target_time)
                    target_schedule.append(
                        {
                            "step": len(target_times),
                            "timestamp": benchmark.format_timestamp(target_time),
                            "time": target_time,
                        }
                    )

            n_forecast = len(actual_scaled)
            if n_forecast == 0:
                continue

            series_data = [
                {
                    "timestamp": benchmark.format_timestamp(timestamp),
                    "value": round_prompt_value(value, benchmark.name),
                }
                for timestamp, value in zip(history_times, history_raw)
            ]

            prompt_payload: dict[str, object] = {
                "symbol": variable_name,
                "interval": "irregular",
                "time_unit": benchmark.time_unit,
                "series": series_data,
                "target_timestamps": target_schedule,
            }
            anchor_predictions_raw: list[float] | None = None

            if benchmark.name == "USHCN":
                prompt_payload["recent_multichannel_tail"] = (
                    build_recent_multichannel_tail(
                        t_input,
                        x_input,
                        columns,
                        benchmark,
                    )
                )
                if config.prompt_style == "gpt_climate_anchor_blend_mse":
                    forecast_anchors = build_ushcn_forecast_anchors(
                        variable_name,
                        history_times,
                        history_raw,
                        target_times,
                    )
                    anchor_predictions_raw = [
                        float(anchor["recommended_anchor"]) for anchor in forecast_anchors
                    ]
                    prompt_payload["forecast_anchors"] = forecast_anchors
            elif benchmark.name == "HumanActivity":
                prompt_payload["recent_multichannel_tail"] = build_recent_multichannel_tail(
                    t_input,
                    x_input,
                    columns,
                    benchmark,
                )
                if config.prompt_style == "gpt_activity_anchor_blend_mse":
                    forecast_anchors = build_activity_forecast_anchors(
                        history_times,
                        history_raw,
                        target_times,
                    )
                    anchor_predictions_raw = [
                        float(anchor["recommended_anchor"]) for anchor in forecast_anchors
                    ]
                    prompt_payload["forecast_anchors"] = forecast_anchors
            elif config.prompt_style == "gpt_clinical_anchor_blend_mse":
                forecast_anchors = build_p12_forecast_anchors(history_raw, target_times)
                anchor_predictions_raw = [
                    float(anchor["recommended_anchor"]) for anchor in forecast_anchors
                ]
                prompt_payload["forecast_anchors"] = forecast_anchors

            history_json = json.dumps(prompt_payload, indent=2)

            valid_sample_count += 1
            trace_sample = should_trace_sample(valid_sample_count, config)
            system_prompt, user_prompt = build_prompt(
                variable_name,
                n_forecast,
                history_json,
                config,
            )

            predictions_raw: list[float] | None = None
            raw_output = ""
            request_start = time.time()
            if trace_sample:
                logger.info(
                    "    [INPUT] sample=%s/%s valid=%s %s=%s variable=%s "
                    "history_n=%s target_n=%s target_times=%s history_tail=%s",
                    sample_index + 1,
                    n_samples,
                    valid_sample_count,
                    benchmark.entity_label,
                    sample.key,
                    variable_name,
                    len(history_raw),
                    n_forecast,
                    ", ".join(benchmark.format_tail_time(value) for value in target_times),
                    format_history_tail(
                        history_times,
                        history_raw,
                        config.trace_history_points,
                        benchmark.format_tail_time,
                    ),
                )

            for attempt in range(config.max_retries):
                if trace_sample:
                    logger.info(
                        "    [CALL] sample=%s %s=%s attempt=%s/%s "
                        "forecast_n=%s prompt_chars=%s",
                        sample_index + 1,
                        benchmark.entity_label,
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
                            "    [OUTPUT] sample=%s %s=%s attempt=%s empty response",
                            sample_index + 1,
                            benchmark.entity_label,
                            sample.key,
                            attempt + 1,
                        )
                    time.sleep(2)
                    continue

                if trace_sample:
                    logger.info(
                        "    [OUTPUT] sample=%s %s=%s attempt=%s raw=%r",
                        sample_index + 1,
                        benchmark.entity_label,
                        sample.key,
                        attempt + 1,
                        raw_output[: config.trace_output_chars],
                    )

                predictions_raw = parse_numbers(raw_output, n_forecast)
                if predictions_raw is not None:
                    if trace_sample:
                        logger.info(
                            "    [PARSE] sample=%s %s=%s parsed_raw=%s",
                            sample_index + 1,
                            benchmark.entity_label,
                            sample.key,
                            format_values(predictions_raw),
                        )
                    break
                if trace_sample:
                    logger.info(
                        "    [PARSE] sample=%s %s=%s failed to parse enough numbers",
                        sample_index + 1,
                        benchmark.entity_label,
                        sample.key,
                    )

            if predictions_raw is None:
                predictions_raw = [history_raw[-1]] * n_forecast
                fallback_count += 1
                if trace_sample or fallback_count <= 5:
                    logger.info(
                        "    [FALLBACK] %s=%s, output=%r",
                        benchmark.entity_label,
                        sample.key,
                        raw_output[:100],
                    )

            gpt_predictions_raw = list(predictions_raw)
            blend_gpt_predictions_raw: list[float] | None = None
            if (
                anchor_predictions_raw is not None
                and config.prompt_style in ANCHOR_BLEND_PROMPT_STYLES
            ):
                blend_gpt_predictions_raw = clamp_to_recent_history(
                    gpt_predictions_raw,
                    history_raw,
                )
                anchor_weight = min(1.0, max(0.0, config.anchor_blend_weight))
                predictions_raw = [
                    anchor_weight * anchor + (1.0 - anchor_weight) * gpt
                    for anchor, gpt in zip(anchor_predictions_raw, blend_gpt_predictions_raw)
                ]
                if trace_sample:
                    logger.info(
                        "    [ANCHOR_BLEND] sample=%s %s=%s weight=%.2f gpt=%s "
                        "clamped_gpt=%s anchor=%s final=%s",
                        sample_index + 1,
                        benchmark.entity_label,
                        sample.key,
                        anchor_weight,
                        format_values(gpt_predictions_raw, digits=4),
                        format_values(blend_gpt_predictions_raw, digits=4),
                        format_values(anchor_predictions_raw, digits=4),
                        format_values(predictions_raw, digits=4),
                    )
            else:
                unclamped_predictions_raw = list(predictions_raw)
                predictions_raw = clamp_to_recent_history(predictions_raw, history_raw)
                if trace_sample and predictions_raw != unclamped_predictions_raw:
                    logger.info(
                        "    [CLAMP] sample=%s %s=%s raw=%s clamped=%s",
                        sample_index + 1,
                        benchmark.entity_label,
                        sample.key,
                        format_values(unclamped_predictions_raw),
                        format_values(predictions_raw),
                    )

            sample_scaled_errors = []
            sample_raw_errors = []
            for step_index in range(n_forecast):
                predicted_scaled = benchmark.raw_to_scaled(
                    predictions_raw[step_index],
                    encoder_index,
                )
                scaled_error = abs(actual_scaled[step_index] - predicted_scaled)

                mae_scaled.append(scaled_error)
                mse_scaled.append((actual_scaled[step_index] - predicted_scaled) ** 2)
                raw_error = abs(actual_raw[step_index] - predictions_raw[step_index])
                mae_raw.append(raw_error)
                sample_scaled_errors.append(scaled_error)
                sample_raw_errors.append(raw_error)

                detail_row = {
                    "Dataset": benchmark.name,
                    "Variable": variable_name,
                    "Entity": sample.key,
                    "Step": step_index,
                    "Actual_scaled": round(actual_scaled[step_index], 6),
                    "Predicted_scaled": round(predicted_scaled, 6),
                    "Actual_raw": round(actual_raw[step_index], 4),
                    "Predicted_raw": round(predictions_raw[step_index], 4),
                    "AE_scaled": round(scaled_error, 6),
                }
                detail_row["Target_time"] = round(target_times[step_index], 2)
                if anchor_predictions_raw is not None:
                    anchor_scaled = benchmark.raw_to_scaled(
                        anchor_predictions_raw[step_index],
                        encoder_index,
                    )
                    gpt_scaled = benchmark.raw_to_scaled(
                        gpt_predictions_raw[step_index],
                        encoder_index,
                    )
                    detail_row["GPT_predicted_scaled"] = round(gpt_scaled, 6)
                    detail_row["GPT_predicted_raw"] = round(gpt_predictions_raw[step_index], 4)
                    if blend_gpt_predictions_raw is not None:
                        blend_gpt_scaled = benchmark.raw_to_scaled(
                            blend_gpt_predictions_raw[step_index],
                            encoder_index,
                        )
                        detail_row["Blend_GPT_predicted_scaled"] = round(blend_gpt_scaled, 6)
                        detail_row["Blend_GPT_predicted_raw"] = round(
                            blend_gpt_predictions_raw[step_index],
                            4,
                        )
                    detail_row["Anchor_scaled"] = round(anchor_scaled, 6)
                    detail_row["Anchor_raw"] = round(anchor_predictions_raw[step_index], 4)
                    detail_row["Anchor_blend_weight"] = round(config.anchor_blend_weight, 3)
                detail_rows.append(detail_row)
                total_predictions += 1

            if trace_sample:
                logger.info(
                    "    [SCORE] sample=%s %s=%s actual_raw=%s predicted_raw=%s "
                    "mae_scaled=%.4f mae_raw=%.4f latency=%.1fs",
                    sample_index + 1,
                    benchmark.entity_label,
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
            "Dataset": benchmark.name,
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
    global_metrics = compute_global_scaled_metrics(config.detail_log_csv)
    total_predictions = results["Predictions"].sum()
    total_fallback = results["Fallback"].sum()

    logger.info("\n%s", "=" * 70)
    logger.info("  %s vs APN - %s results", config.model_id, benchmark.display_name)
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
    logger.info("  Equal-variable average:")
    logger.info("    MAE_scaled = %.6f", avg_mae)
    logger.info("    MSE_scaled = %.6f", avg_mse)
    if global_metrics is not None:
        global_mae, global_mse = global_metrics
        logger.info("  APN-style global masked metrics:")
        logger.info("    MAE_scaled = %.6f", global_mae)
        logger.info("    MSE_scaled = %.6f", global_mse)
    logger.info("    Total predictions: %s", f"{total_predictions:,}")
    logger.info("    Total fallback: %s", total_fallback)
    logger.info("    Total time: %.0fs (%.1fh)", total_elapsed, total_elapsed / 3600)
    baseline = APN_BASELINES.get(benchmark.name)
    if baseline is not None:
        logger.info(
            "  APN paper baseline: MAE=%.4f, MSE=%.4f",
            baseline["MAE"],
            baseline["MSE"],
        )

    comparison_mae = global_metrics[0] if global_metrics is not None else avg_mae
    comparison_mse = global_metrics[1] if global_metrics is not None else avg_mse
    logger.info(
        "  %s comparison result: MAE=%.4f, MSE=%.4f",
        config.model_id,
        comparison_mae,
        comparison_mse,
    )

    if baseline is not None:
        mae_delta = baseline["MAE"] - comparison_mae
        mse_delta = baseline["MSE"] - comparison_mse
        logger.info(
            "  %s delta vs APN: MAE=%+.4f, MSE=%+.4f",
            benchmark.name,
            mae_delta,
            mse_delta,
        )

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
