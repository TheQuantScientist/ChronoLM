"""Anchor-only baselines on APN irregular time-series benchmark splits."""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

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

APN_BASELINES = {
    "HumanActivity": {"MAE": 0.1159, "MSE": 0.0421},
    "P12": {"MAE": 0.3650, "MSE": 0.3093},
    "USHCN": {"MAE": 0.2611, "MSE": 0.1590},
}

COMMON_ANCHOR_METHODS = [
    "last",
    "mean2",
    "mean3",
    "mean5",
    "trim3",
    "trim5",
    "ema01",
    "ema02",
    "ema03",
    "ema04",
    "ema05",
    "ema06",
    "ema07",
    "ema08",
    "ema09",
]

DATASET_ANCHOR_METHODS = {
    "P12": COMMON_ANCHOR_METHODS,
    "HumanActivity": COMMON_ANCHOR_METHODS,
    "USHCN": COMMON_ANCHOR_METHODS
    + [
        "mode",
        "phase",
        "trend",
        "last80_phase20",
        "phase75_last25",
        "phase65_trend24_last11",
        "phase50_last50",
    ],
}


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
class AnchorConfig:
    """Configuration for one deterministic anchor benchmark run."""

    dataset_name: str = "P12"
    seq_len: int | None = None
    pred_len: int | None = None
    max_test_samples: int | None = None
    run_name: str | None = None
    output_dir: Path | None = None
    output_csv: Path | None = None
    checkpoint_csv: Path | None = None
    detail_log_csv: Path | None = None
    calibration_csv: Path | None = None
    debug_log: Path | None = None
    trace_every: int = 0
    progress_every: int = 100
    trace_history_points: int = 5

    def __post_init__(self) -> None:
        dataset_name = canonical_dataset_name(self.dataset_name)
        defaults = DATASET_DEFAULTS[dataset_name]
        run_name = self.run_name or f"anchor_only_{dataset_name.lower()}"
        output_dir = self.output_dir or Path("anchor_results") / dataset_name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "seq_len", self.seq_len or defaults["seq_len"])
        object.__setattr__(self, "pred_len", self.pred_len or defaults["pred_len"])
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "output_dir", output_dir)

        def default_path(path: Path | None, suffix: str) -> Path:
            return path or output_dir / f"{run_name}_{dataset_name}_{suffix}"

        object.__setattr__(self, "output_csv", default_path(self.output_csv, "Results.csv"))
        object.__setattr__(
            self,
            "checkpoint_csv",
            default_path(self.checkpoint_csv, "Checkpoint.csv"),
        )
        object.__setattr__(
            self,
            "detail_log_csv",
            default_path(self.detail_log_csv, "DetailLog.csv"),
        )
        object.__setattr__(
            self,
            "calibration_csv",
            default_path(self.calibration_csv, "Calibration.csv"),
        )
        object.__setattr__(self, "debug_log", default_path(self.debug_log, "debug.log"))


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


def most_common_rounded(values: list[float], digits: int = 4) -> float:
    rounded_values = [round(float(value), digits) for value in values]
    return float(Counter(rounded_values).most_common(1)[0][0])


def clamp_to_recent_history(values: Iterable[float], history: list[float]) -> list[float]:
    if len(history) < 3:
        return list(values)

    recent = history[-10:]
    mean = float(np.mean(recent))
    std = max(float(np.std(recent)), 0.1)
    lower = mean - 3 * std
    upper = mean + 3 * std
    return [max(lower, min(upper, float(value))) for value in values]


def should_trace_sample(valid_sample_count: int, config: AnchorConfig) -> bool:
    return config.trace_every > 0 and valid_sample_count % config.trace_every == 0


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


def p12_scaled_anchor_base(method: str, history_scaled: list[float]) -> float:
    values = np.asarray(history_scaled, dtype=float)

    if method == "last":
        return float(values[-1])

    if method.startswith("mean"):
        count = min(int(method[-1]), len(values))
        return float(np.mean(values[-count:]))

    if method.startswith("trim"):
        count = min(int(method[-1]), len(values))
        recent = np.sort(values[-count:])
        if len(recent) >= 3:
            return float(np.mean(recent[1:-1]))
        return float(np.mean(recent))

    if method.startswith("ema"):
        alpha = int(method[-2:]) / 10.0
        anchor = float(values[0])
        for value in values[1:]:
            anchor = alpha * float(value) + (1.0 - alpha) * anchor
        return anchor

    raise ValueError(f"Unknown P12 anchor method: {method}")


def simple_anchor_base(
    method: str,
    history_times: list[float],
    history_values: list[float],
    target_time: float,
) -> float:
    values = np.asarray(history_values, dtype=float)

    if method == "mode":
        return most_common_rounded(history_values)

    if method == "phase":
        return ushcn_annual_phase_anchor(history_times, history_values, target_time)

    if method == "trend":
        return ushcn_recent_trend_anchor(history_times, history_values, target_time)

    if method == "last80_phase20":
        last_value = float(history_values[-1])
        phase_value = ushcn_annual_phase_anchor(history_times, history_values, target_time)
        return 0.8 * last_value + 0.2 * phase_value

    if method == "phase75_last25":
        last_value = float(history_values[-1])
        phase_value = ushcn_annual_phase_anchor(history_times, history_values, target_time)
        return 0.75 * phase_value + 0.25 * last_value

    if method == "phase65_trend24_last11":
        last_value = float(history_values[-1])
        phase_value = ushcn_annual_phase_anchor(history_times, history_values, target_time)
        trend_value = ushcn_recent_trend_anchor(history_times, history_values, target_time)
        return 0.65 * phase_value + 0.24 * trend_value + 0.11 * last_value

    if method == "phase50_last50":
        last_value = float(history_values[-1])
        phase_value = ushcn_annual_phase_anchor(history_times, history_values, target_time)
        return 0.5 * phase_value + 0.5 * last_value

    if method == "last":
        return float(values[-1])

    if method.startswith("mean"):
        count = min(int(method[-1]), len(values))
        return float(np.mean(values[-count:]))

    if method.startswith("trim"):
        count = min(int(method[-1]), len(values))
        recent = np.sort(values[-count:])
        if len(recent) >= 3:
            return float(np.mean(recent[1:-1]))
        return float(np.mean(recent))

    if method.startswith("ema"):
        alpha = int(method[-2:]) / 10.0
        anchor = float(values[0])
        for value in values[1:]:
            anchor = alpha * float(value) + (1.0 - alpha) * anchor
        return anchor

    raise ValueError(f"Unknown anchor method: {method}")


@dataclass(frozen=True)
class AnchorSpec:
    method: str
    beta: float
    calibration_mae: float
    calibration_mse: float
    calibration_points: int
    source: str = "non_test_mse"
    rationale: str = ""


def anchor_base_forecast_scaled(
    benchmark: "BenchmarkData",
    method: str,
    history_times: list[float],
    history_raw: list[float],
    history_scaled: list[float],
    target_times: list[float],
) -> list[float]:
    if benchmark.name == "P12":
        base_value = p12_scaled_anchor_base(method, history_scaled)
        return [base_value for _target_time in target_times]

    return [
        simple_anchor_base(method, history_times, history_raw, target_time)
        for target_time in target_times
    ]


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    display_name: str
    test_dataset: object
    calibration_dataset: object
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


def materialize_dataset(dataset: object) -> list[object]:
    return [dataset[index] for index in range(len(dataset))]


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


def load_benchmark_data(config: AnchorConfig) -> BenchmarkData:
    if config.dataset_name == "P12":
        from data.dependencies.tsdm.tasks.P12 import Physionet2012

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

        train_dataset = task.get_dataset((0, "train"))
        validation_dataset = task.get_dataset((0, "val"))

        return BenchmarkData(
            name="P12",
            display_name="PhysioNet P12",
            test_dataset=task.get_dataset((0, "test")),
            calibration_dataset=materialize_dataset(train_dataset)
            + materialize_dataset(validation_dataset),
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
        from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

        task = USHCN_DeBrouwer2019(
            normalize_time=False,
            seq_len=config.seq_len - 0.5,
            pred_len=config.pred_len,
        )
        columns = list(task.dataset.columns)

        train_dataset = task.get_dataset((0, "train"))
        validation_dataset = task.get_dataset((0, "val"))

        return BenchmarkData(
            name="USHCN",
            display_name="USHCN",
            test_dataset=task.get_dataset((0, "test")),
            calibration_dataset=materialize_dataset(train_dataset)
            + materialize_dataset(validation_dataset),
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
        from data.dependencies.HumanActivity.HumanActivity import HumanActivity
        from sklearn import model_selection

        dataset_root = Path(
            os.getenv(
                "CHRONOLM_HUMANACTIVITY_ROOT",
                str(APN_ROOT / "storage" / "datasets" / "HumanActivity"),
            )
        )
        processed_path = dataset_root / "processed" / "data.pt"
        human_activity = HumanActivity(root=str(dataset_root), download=not processed_path.exists())
        seen_records, test_records = model_selection.train_test_split(
            human_activity,
            train_size=0.9,
            random_state=42,
            shuffle=False,
        )
        train_records, validation_records = model_selection.train_test_split(
            seen_records,
            train_size=0.9,
            random_state=42,
            shuffle=False,
        )

        tag_names = ["ANKLE_LEFT", "ANKLE_RIGHT", "CHEST", "BELT"]
        axis_names = ["x", "y", "z"]
        columns = [f"{tag}_{axis}" for tag in tag_names for axis in axis_names]

        train_samples = build_human_activity_samples(
            train_records,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
        )
        validation_samples = build_human_activity_samples(
            validation_records,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
        )

        return BenchmarkData(
            name="HumanActivity",
            display_name="HumanActivity",
            test_dataset=build_human_activity_samples(
                test_records,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
            ),
            calibration_dataset=train_samples + validation_samples,
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


def compute_global_scaled_metrics(detail_log_csv: Path) -> dict[str, float] | None:
    if not detail_log_csv.exists():
        return None

    detail = pd.read_csv(detail_log_csv)
    required = {"Actual_scaled", "Predicted_scaled"}
    if not required <= set(detail.columns):
        return None

    valid = detail[["Actual_scaled", "Predicted_scaled"]].notna().all(axis=1)
    detail = detail.loc[valid].copy()
    if detail.empty:
        return None

    residual = detail["Actual_scaled"] - detail["Predicted_scaled"]
    detail["AE_scaled"] = residual.abs()
    detail["SE_scaled"] = residual**2
    per_variable = (
        detail.groupby("Variable", sort=False)
        .agg(MAE_scaled=("AE_scaled", "mean"), MSE_scaled=("SE_scaled", "mean"))
        .reset_index()
    )

    return {
        "MAE_scaled": float(detail["AE_scaled"].mean()),
        "MSE_scaled": float(detail["SE_scaled"].mean()),
        "Variable_avg_MAE_scaled": float(per_variable["MAE_scaled"].mean()),
        "Variable_avg_MSE_scaled": float(per_variable["MSE_scaled"].mean()),
        "Predictions": float(len(detail)),
    }


def extract_variable_sample(
    sample: object,
    benchmark: BenchmarkData,
    variable_index: int,
    encoder_index: int,
    pred_len: int,
) -> dict[str, object] | None:
    t_input, x_input, t_target = sample.inputs
    y_target = sample.targets

    history_scaled_tensor = x_input[:, variable_index]
    target_scaled_tensor = y_target[:, variable_index]
    valid_history_mask = ~history_scaled_tensor.isnan()
    valid_target_mask = ~target_scaled_tensor.isnan()

    if valid_history_mask.sum() < 3 or valid_target_mask.sum() == 0:
        return None

    valid_history = history_scaled_tensor[valid_history_mask]
    valid_history_t = t_input[valid_history_mask]
    history_scaled = [float(value.item()) for value in valid_history]
    history_raw = [
        benchmark.scaled_to_raw(float(value.item()), encoder_index)
        for value in valid_history
    ]
    history_times = [
        round(benchmark.time_to_prompt(float(value.item())), 2)
        for value in valid_history_t
    ]

    actual_scaled: list[float] = []
    actual_raw: list[float] = []
    target_times: list[float] = []
    forecast_limit = min(pred_len, len(target_scaled_tensor), len(t_target))
    for step_index in range(forecast_limit):
        if valid_target_mask[step_index]:
            target_value = float(target_scaled_tensor[step_index].item())
            target_time = round(
                benchmark.time_to_prompt(float(t_target[step_index].item())),
                2,
            )
            actual_scaled.append(target_value)
            actual_raw.append(benchmark.scaled_to_raw(target_value, encoder_index))
            target_times.append(target_time)

    if not actual_scaled:
        return None

    return {
        "history_scaled": history_scaled,
        "history_raw": history_raw,
        "history_times": history_times,
        "actual_scaled": actual_scaled,
        "actual_raw": actual_raw,
        "target_times": target_times,
    }


def fit_beta(base_values: list[float], target_values: list[float]) -> float:
    x = np.asarray(base_values, dtype=float)
    y = np.asarray(target_values, dtype=float)
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 1.0
    beta = float(np.dot(x, y) / denominator)
    return float(np.clip(beta, 0.0, 1.25))


def collect_method_values(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    variable_index: int,
    encoder_index: int,
    method: str,
) -> tuple[list[float], list[float]]:
    base_values: list[float] = []
    target_values: list[float] = []

    for sample_index in range(len(benchmark.calibration_dataset)):
        sample = benchmark.calibration_dataset[sample_index]
        extracted = extract_variable_sample(
            sample=sample,
            benchmark=benchmark,
            variable_index=variable_index,
            encoder_index=encoder_index,
            pred_len=config.pred_len,
        )
        if extracted is None:
            continue

        base_scaled = anchor_base_forecast_scaled(
            benchmark=benchmark,
            method=method,
            history_times=extracted["history_times"],
            history_raw=extracted["history_raw"],
            history_scaled=extracted["history_scaled"],
            target_times=extracted["target_times"],
        )
        base_values.extend(base_scaled)
        target_values.extend(extracted["actual_scaled"])

    return base_values, target_values


def score_anchor_spec(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    variable_index: int,
    encoder_index: int,
    method: str,
    beta: float,
) -> tuple[float, float, int]:
    base_values, target_values = collect_method_values(
        benchmark=benchmark,
        config=config,
        variable_index=variable_index,
        encoder_index=encoder_index,
        method=method,
    )
    if not base_values:
        return float("inf"), float("inf"), 0

    residual = np.asarray(target_values, dtype=float) - beta * np.asarray(
        base_values,
        dtype=float,
    )
    mae = float(np.abs(residual).mean())
    mse = float((residual**2).mean())
    return mae, mse, len(base_values)


def append_calibration_row(
    rows: list[dict[str, object]],
    benchmark: BenchmarkData,
    variable_name: str,
    spec: AnchorSpec,
) -> None:
    rows.append(
        {
            "Dataset": benchmark.name,
            "Variable": variable_name,
            "Method": spec.method,
            "Beta": round(spec.beta, 6),
            "Calibration_MAE_scaled": round(spec.calibration_mae, 6),
            "Calibration_MSE_scaled": round(spec.calibration_mse, 6),
            "Calibration_points": spec.calibration_points,
            "Source": spec.source,
            "Rationale": spec.rationale,
        }
    )


def calibrate_by_non_test_mse(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> dict[str, AnchorSpec]:
    methods = DATASET_ANCHOR_METHODS[benchmark.name]
    specs: dict[str, AnchorSpec] = {}
    for variable_index, variable_name in enumerate(benchmark.columns):
        encoder_index = benchmark.encoder_index_by_column[variable_name]
        best_method = "last"
        best_beta = 1.0
        best_mae = float("inf")
        best_mse = float("inf")
        best_points = 0

        for method in methods:
            base_values, target_values = collect_method_values(
                benchmark=benchmark,
                config=config,
                variable_index=variable_index,
                encoder_index=encoder_index,
                method=method,
            )
            if not base_values:
                continue

            beta = fit_beta(base_values, target_values)
            residual = np.asarray(target_values, dtype=float) - beta * np.asarray(
                base_values,
                dtype=float,
            )
            mae = float(np.abs(residual).mean())
            mse = float((residual**2).mean())
            if (mse, mae, method) < (best_mse, best_mae, best_method):
                best_method = method
                best_beta = beta
                best_mae = mae
                best_mse = mse
                best_points = len(base_values)

        spec = AnchorSpec(
            method=best_method,
            beta=best_beta,
            calibration_mae=best_mae,
            calibration_mse=best_mse,
            calibration_points=best_points,
            source="non_test_mse",
            rationale="method and beta selected by lowest scaled MSE on APN non-test split",
        )
        specs[variable_name] = spec
        append_calibration_row(rows, benchmark, variable_name, spec)
        logger.info(
            "  [CALIBRATE] %s method=%s beta=%.4f cal_mae=%.6f cal_mse=%.6f points=%s",
            variable_name,
            spec.method,
            spec.beta,
            spec.calibration_mae,
            spec.calibration_mse,
            spec.calibration_points,
        )

    return specs


def ushcn_history_statistics(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    variable_index: int,
    encoder_index: int,
) -> dict[str, float]:
    values: list[float] = []
    phase_last: list[float] = []
    phase_trend: list[float] = []

    for sample in benchmark.calibration_dataset:
        extracted = extract_variable_sample(
            sample=sample,
            benchmark=benchmark,
            variable_index=variable_index,
            encoder_index=encoder_index,
            pred_len=config.pred_len,
        )
        if extracted is None:
            continue
        values.extend(extracted["history_raw"])
        for target_time in extracted["target_times"]:
            phase_value = ushcn_annual_phase_anchor(
                extracted["history_times"],
                extracted["history_raw"],
                target_time,
            )
            trend_value = ushcn_recent_trend_anchor(
                extracted["history_times"],
                extracted["history_raw"],
                target_time,
            )
            last_value = float(extracted["history_raw"][-1])
            phase_last.append(abs(phase_value - last_value))
            phase_trend.append(abs(phase_value - trend_value))

    if not values:
        return {
            "mode_fraction": 0.0,
            "unique_count": 0.0,
            "phase_last_gap": 0.0,
            "phase_trend_gap": 0.0,
        }

    rounded = [round(float(value), 4) for value in values]
    counts = Counter(rounded)
    mode_count = counts.most_common(1)[0][1]
    return {
        "mode_fraction": mode_count / len(rounded),
        "unique_count": float(len(counts)),
        "phase_last_gap": float(np.mean(phase_last)) if phase_last else 0.0,
        "phase_trend_gap": float(np.mean(phase_trend)) if phase_trend else 0.0,
    }


def choose_ushcn_structural_method(stats: dict[str, float]) -> tuple[str, str]:
    mode_fraction = stats["mode_fraction"]
    unique_count = stats["unique_count"]

    if mode_fraction > 0.95:
        return "mode", "dominant sparse value covers more than 95% of observed history"

    if mode_fraction > 0.50 and unique_count > 150:
        return "mode", "sparse channel with high-cardinality events and dominant baseline"

    if mode_fraction > 0.50:
        return "last80_phase20", "sparse channel with meaningful recent departures from baseline"

    if stats["phase_trend_gap"] <= stats["phase_last_gap"]:
        return "phase65_trend24_last11", "dense seasonal channel where trend tracks phase"

    return "phase75_last25", "dense seasonal channel where phase is cleaner than trend"


def calibrate_ushcn_structural(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> dict[str, AnchorSpec]:
    specs: dict[str, AnchorSpec] = {}

    for variable_index, variable_name in enumerate(benchmark.columns):
        encoder_index = benchmark.encoder_index_by_column[variable_name]
        stats = ushcn_history_statistics(benchmark, config, variable_index, encoder_index)
        method, rationale = choose_ushcn_structural_method(stats)
        beta = 1.0
        mae, mse, points = score_anchor_spec(
            benchmark=benchmark,
            config=config,
            variable_index=variable_index,
            encoder_index=encoder_index,
            method=method,
            beta=beta,
        )
        spec = AnchorSpec(
            method=method,
            beta=beta,
            calibration_mae=mae,
            calibration_mse=mse,
            calibration_points=points,
            source="history_structure",
            rationale=(
                f"{rationale}; mode_fraction={stats['mode_fraction']:.4f}; "
                f"unique_count={stats['unique_count']:.0f}; "
                f"phase_last_gap={stats['phase_last_gap']:.4f}; "
                f"phase_trend_gap={stats['phase_trend_gap']:.4f}"
            ),
        )
        specs[variable_name] = spec
        append_calibration_row(rows, benchmark, variable_name, spec)
        logger.info(
            "  [STRUCTURE] %s method=%s beta=%.1f cal_mae=%.6f cal_mse=%.6f points=%s",
            variable_name,
            spec.method,
            spec.beta,
            spec.calibration_mae,
            spec.calibration_mse,
            spec.calibration_points,
        )

    return specs


def calibrate_human_activity_structural(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> dict[str, AnchorSpec]:
    horizon_ratio = float(config.pred_len) / float(config.seq_len)
    alpha = float(np.clip(round(np.sqrt(horizon_ratio) * 10) / 10, 0.1, 0.9))
    method = f"ema{int(alpha * 10):02d}"
    specs: dict[str, AnchorSpec] = {}

    for variable_index, variable_name in enumerate(benchmark.columns):
        encoder_index = benchmark.encoder_index_by_column[variable_name]
        mae, mse, points = score_anchor_spec(
            benchmark=benchmark,
            config=config,
            variable_index=variable_index,
            encoder_index=encoder_index,
            method=method,
            beta=1.0,
        )
        spec = AnchorSpec(
            method=method,
            beta=1.0,
            calibration_mae=mae,
            calibration_mse=mse,
            calibration_points=points,
            source="horizon_structure",
            rationale=(
                f"EMA alpha rounded from sqrt(pred_len/seq_len)={np.sqrt(horizon_ratio):.4f}"
            ),
        )
        specs[variable_name] = spec
        append_calibration_row(rows, benchmark, variable_name, spec)
        logger.info(
            "  [STRUCTURE] %s method=%s beta=1.0 cal_mae=%.6f cal_mse=%.6f points=%s",
            variable_name,
            spec.method,
            spec.calibration_mae,
            spec.calibration_mse,
            spec.calibration_points,
        )

    return specs


def calibrate_anchor_specs(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
) -> dict[str, AnchorSpec]:
    rows: list[dict[str, object]] = []

    logger.info("Calibrating anchor methods on APN non-test split...")
    logger.info("  Calibration samples: %s", len(benchmark.calibration_dataset))

    if benchmark.name == "USHCN":
        logger.info("  Calibration source: observed-history structure, no target labels")
        specs = calibrate_ushcn_structural(benchmark, config, logger, rows)
    elif benchmark.name == "HumanActivity":
        logger.info("  Calibration source: forecast horizon structure, no target labels")
        specs = calibrate_human_activity_structural(benchmark, config, logger, rows)
    else:
        methods = DATASET_ANCHOR_METHODS[benchmark.name]
        logger.info("  Calibration source: non-test target labels")
        logger.info("  Candidate methods: %s", ", ".join(methods))
        specs = calibrate_by_non_test_mse(benchmark, config, logger, rows)

    pd.DataFrame(rows).to_csv(config.calibration_csv, index=False)
    logger.info("Calibration CSV: %s", config.calibration_csv)
    return specs


def build_forecast_anchors(
    benchmark: BenchmarkData,
    history_times: list[float],
    history_raw: list[float],
    history_scaled: list[float],
    target_times: list[float],
    encoder_index: int,
    spec: AnchorSpec,
) -> list[dict[str, float | int | str]]:
    base_scaled_values = anchor_base_forecast_scaled(
        benchmark=benchmark,
        method=spec.method,
        history_times=history_times,
        history_raw=history_raw,
        history_scaled=history_scaled,
        target_times=target_times,
    )

    anchors: list[dict[str, float | int | str]] = []
    for step_index, (target_time, base_scaled) in enumerate(
        zip(target_times, base_scaled_values),
        start=1,
    ):
        anchor_scaled = spec.beta * float(base_scaled)
        anchor_raw = benchmark.scaled_to_raw(anchor_scaled, encoder_index)
        anchors.append(
            {
                "step": step_index,
                "target_time": round(float(target_time), 2),
                "recommended_anchor": round(float(anchor_raw), 4),
                "recommended_anchor_scaled": round(anchor_scaled, 6),
                "base_scaled": round(float(base_scaled), 6),
                "method": spec.method,
                "shrink_beta": round(float(spec.beta), 6),
                "calibration_mae": round(float(spec.calibration_mae), 6),
                "calibration_mse": round(float(spec.calibration_mse), 6),
                "rule": spec.source,
            }
        )

    return anchors



def run(config: AnchorConfig) -> dict[str, float | str]:
    warnings.filterwarnings("ignore")
    logger = configure_logging(config.debug_log)

    logger.info("Run name: %s", config.run_name)
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Method: anchor_only")
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Loading %s data through the APN pipeline...", config.dataset_name)
    load_start = time.time()
    benchmark = load_benchmark_data(config)
    columns = benchmark.columns
    test_dataset = benchmark.test_dataset
    total_test_samples = len(test_dataset)
    n_samples = min(config.max_test_samples or total_test_samples, total_test_samples)

    logger.info("  Data loaded in %.1fs", time.time() - load_start)
    logger.info("  Test samples: %s, using: %s", total_test_samples, n_samples)
    logger.info("  Variables: %s", len(columns))
    anchor_specs = calibrate_anchor_specs(benchmark, config, logger)

    all_results, processed_variables = load_checkpoint(config.checkpoint_csv, logger)

    logger.info("\n%s", "=" * 70)
    logger.info("  Anchor-only baseline - %s", benchmark.display_name)
    logger.info("  Lookback=%s, forecast=%s", config.seq_len, config.pred_len)
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
            sample_key = getattr(sample, "key", sample_index)
            extracted = extract_variable_sample(
                sample=sample,
                benchmark=benchmark,
                variable_index=variable_index,
                encoder_index=encoder_index,
                pred_len=config.pred_len,
            )
            if extracted is None:
                continue

            history_scaled_values = extracted["history_scaled"]
            history_raw = extracted["history_raw"]
            history_times = extracted["history_times"]
            actual_scaled = extracted["actual_scaled"]
            actual_raw = extracted["actual_raw"]
            target_times = extracted["target_times"]
            n_forecast = len(actual_scaled)
            valid_sample_count += 1
            trace_sample = should_trace_sample(valid_sample_count, config)
            request_start = time.time()

            forecast_anchors = build_forecast_anchors(
                benchmark=benchmark,
                history_times=history_times,
                history_raw=history_raw,
                history_scaled=history_scaled_values,
                target_times=target_times,
                encoder_index=encoder_index,
                spec=anchor_specs[variable_name],
            )
            anchor_predictions_raw = [
                float(anchor["recommended_anchor"]) for anchor in forecast_anchors
            ]
            predictions_raw = clamp_to_recent_history(anchor_predictions_raw, history_raw)

            if trace_sample:
                logger.info(
                    "    [INPUT] sample=%s/%s valid=%s %s=%s variable=%s "
                    "history_n=%s target_n=%s target_times=%s history_tail=%s",
                    sample_index + 1,
                    n_samples,
                    valid_sample_count,
                    benchmark.entity_label,
                    sample_key,
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
                logger.info(
                    "    [ANCHOR] sample=%s %s=%s anchor=%s predicted=%s",
                    sample_index + 1,
                    benchmark.entity_label,
                    sample_key,
                    format_values(anchor_predictions_raw, digits=4),
                    format_values(predictions_raw, digits=4),
                )

            sample_scaled_errors = []
            sample_raw_errors = []
            for step_index in range(n_forecast):
                predicted_scaled = benchmark.raw_to_scaled(
                    predictions_raw[step_index],
                    encoder_index,
                )
                scaled_error = abs(actual_scaled[step_index] - predicted_scaled)
                squared_error = (actual_scaled[step_index] - predicted_scaled) ** 2
                raw_error = abs(actual_raw[step_index] - predictions_raw[step_index])

                mae_scaled.append(scaled_error)
                mse_scaled.append(squared_error)
                mae_raw.append(raw_error)
                sample_scaled_errors.append(scaled_error)
                sample_raw_errors.append(raw_error)

                anchor = forecast_anchors[step_index]
                anchor_scaled = benchmark.raw_to_scaled(
                    anchor_predictions_raw[step_index],
                    encoder_index,
                )
                detail_rows.append(
                    {
                        "Dataset": benchmark.name,
                        "Method": "anchor_only",
                        "Variable": variable_name,
                        "Entity": sample_key,
                        "Step": step_index,
                        "Target_time": round(target_times[step_index], 2),
                        "Actual_scaled": round(actual_scaled[step_index], 6),
                        "Predicted_scaled": round(predicted_scaled, 6),
                        "Actual_raw": round(actual_raw[step_index], 4),
                        "Predicted_raw": round(predictions_raw[step_index], 4),
                        "AE_scaled": round(scaled_error, 6),
                        "SE_scaled": round(squared_error, 6),
                        "Anchor_scaled": round(anchor_scaled, 6),
                        "Anchor_raw": round(anchor_predictions_raw[step_index], 4),
                        "Anchor_rule": anchor.get("rule", ""),
                        "Anchor_method": anchor.get("method", ""),
                        "Anchor_shrink_beta": anchor.get("shrink_beta", ""),
                    }
                )
                total_predictions += 1

            if trace_sample:
                logger.info(
                    "    [SCORE] sample=%s %s=%s actual_raw=%s predicted_raw=%s "
                    "mae_scaled=%.4f mae_raw=%.4f latency=%.1fs",
                    sample_index + 1,
                    benchmark.entity_label,
                    sample_key,
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
            "Method": "anchor_only",
            "MAE_scaled": round(float(np.mean(mae_scaled)), 6),
            "MSE_scaled": round(float(np.mean(mse_scaled)), 6),
            "MAE_raw": round(float(np.mean(mae_raw)), 4),
            "Predictions": total_predictions,
            "Fallback": fallback_count,
            "Time_s": round(variable_elapsed, 1),
        }
        all_results.append(result_row)

        logger.info(
            "\n  %s: MAE_scaled=%s, MSE_scaled=%s, MAE_raw=%s, fallback=%s, time=%ss",
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
        return {
            "Dataset": benchmark.name,
            "Method": "anchor_only",
            "MAE_scaled": float("nan"),
            "MSE_scaled": float("nan"),
        }

    results = pd.DataFrame(all_results)
    results.to_csv(config.output_csv, index=False)

    total_elapsed = time.time() - total_start
    avg_mae = float(results["MAE_scaled"].mean())
    avg_mse = float(results["MSE_scaled"].mean())
    global_metrics = compute_global_scaled_metrics(config.detail_log_csv)
    total_predictions = int(results["Predictions"].sum())
    total_fallback = int(results["Fallback"].sum())

    logger.info("\n%s", "=" * 70)
    logger.info("  Anchor-only baseline - %s results", benchmark.display_name)
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
                "Time_s",
            ]
        ].to_string(index=False),
    )
    logger.info("-" * 70)
    logger.info("Equal-variable MAE_scaled: %.6f", avg_mae)
    logger.info("Equal-variable MSE_scaled: %.6f", avg_mse)
    if global_metrics is not None:
        logger.info("APN-style global MAE_scaled: %.6f", global_metrics["MAE_scaled"])
        logger.info("APN-style global MSE_scaled: %.6f", global_metrics["MSE_scaled"])
    logger.info("Total predictions: %s", total_predictions)
    logger.info("Total fallbacks: %s", total_fallback)
    logger.info("Total time: %.1fs", total_elapsed)

    baseline = APN_BASELINES.get(benchmark.name)
    if baseline is not None and global_metrics is not None:
        mae_delta = baseline["MAE"] - global_metrics["MAE_scaled"]
        mse_delta = baseline["MSE"] - global_metrics["MSE_scaled"]
        logger.info("APN paper baseline: MAE=%.4f, MSE=%.4f", baseline["MAE"], baseline["MSE"])
        logger.info(
            "Anchor-only comparison result: MAE=%.4f, MSE=%.4f",
            global_metrics["MAE_scaled"],
            global_metrics["MSE_scaled"],
        )
        logger.info("%s delta vs APN: MAE=%+.4f, MSE=%+.4f", benchmark.name, mae_delta, mse_delta)

    logger.info("Output CSV: %s", config.output_csv)
    logger.info("Detail log CSV: %s", config.detail_log_csv)
    logger.info("Debug log: %s", config.debug_log)

    summary = {
        "Dataset": benchmark.name,
        "Method": "anchor_only",
        "Equal_variable_MAE_scaled": avg_mae,
        "Equal_variable_MSE_scaled": avg_mse,
        "Predictions": float(total_predictions),
        "Fallback": float(total_fallback),
        "Time_s": total_elapsed,
    }
    if global_metrics is not None:
        summary.update(
            {
                "MAE_scaled": global_metrics["MAE_scaled"],
                "MSE_scaled": global_metrics["MSE_scaled"],
            }
        )
    return summary
