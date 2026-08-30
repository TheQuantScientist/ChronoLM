"""AutoAnchor baselines on APN irregular time-series benchmark splits."""

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


# APN protocol metadata. These values define the public benchmark windows;
# they are not AutoAnchor tuning knobs.
DATASET_DEFAULTS = {
    "P12": {"seq_len": 36, "pred_len": 3, "display_name": "PhysioNet P12"},
    "USHCN": {"seq_len": 150, "pred_len": 3, "display_name": "USHCN"},
    "HumanActivity": {
        "seq_len": 3000,
        "pred_len": 300,
        "display_name": "HumanActivity",
    },
}

# Paper numbers used only for logging deltas after evaluation.
APN_BASELINES = {
    "HumanActivity": {"MAE": 0.1159, "MSE": 0.0421},
    "P12": {"MAE": 0.3650, "MSE": 0.3093},
    "USHCN": {"MAE": 0.2611, "MSE": 0.1590},
}

DEFAULT_ANCHOR_METHOD = "AutoAnchor"
ANCHOR_FAMILY_METHODS = (
    "NaiveAnchor",
    "ExpoAnchor",
    "SparseAnchor",
    "ERMAnchor",
    "AutoAnchor",
)
ANCHOR_METHOD_SLUGS = {
    "NaiveAnchor": "naive_anchor",
    "ExpoAnchor": "expo_anchor",
    "SparseAnchor": "sparse_anchor",
    "ERMAnchor": "erm_anchor",
    "AutoAnchor": "auto_anchor",
}
RUN_NAME_PREFIX = "anchor_family"

PHASE_COMPONENTS = {
    "phase025k3": (0.25, 3),
    "phase033k5": (1.0 / 3.0, 5),
    "phase050k5": (0.50, 5),
    "phase100k7": (1.00, 7),
}


@dataclass(frozen=True)
class AnchorCandidate:
    """One candidate forecast rule in the shared AutoAnchor library."""

    name: str
    components: tuple[tuple[str, float], ...]


def weight_tag(weight: float) -> str:
    return f"{int(round(weight * 100)):02d}"


def build_auto_anchor_candidates() -> list[AnchorCandidate]:
    """Build the same finite anchor library for every dataset and variable."""
    base_components = [
        "last",
        "mean2",
        "mean3",
        "mean5",
        "mean8",
        "trim3",
        "trim5",
        "trim8",
        "mode",
        "trend3",
        "trend5",
        "trend8",
        "trend12",
        *[f"ema{alpha:02d}" for alpha in range(1, 10)],
        *PHASE_COMPONENTS.keys(),
    ]
    candidates = [
        AnchorCandidate(component, ((component, 1.0),))
        for component in base_components
    ]

    pair_components = [
        ("last", "phase033k5"),
        ("phase033k5", "trend8"),
        ("last", "trend8"),
        ("ema03", "last"),
    ]
    pair_weights = sorted({*[index / 10.0 for index in range(1, 10)], 0.25})
    for first, second in pair_components:
        for first_weight in pair_weights:
            second_weight = 1.0 - first_weight
            name = (
                f"mix_{first}{weight_tag(first_weight)}_"
                f"{second}{weight_tag(second_weight)}"
            )
            candidates.append(
                AnchorCandidate(
                    name,
                    ((first, first_weight), (second, second_weight)),
                )
            )

    phase_weights = [0.5, 0.6, 0.65, 0.7, 0.8]
    trend_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
    for phase_weight in phase_weights:
        for trend_weight in trend_weights:
            last_weight = 1.0 - phase_weight - trend_weight
            if last_weight < 0.05:
                continue
            name = (
                f"mix_phase033k5{weight_tag(phase_weight)}_"
                f"trend8{weight_tag(trend_weight)}_"
                f"last{weight_tag(last_weight)}"
            )
            candidates.append(
                AnchorCandidate(
                    name,
                    (
                        ("phase033k5", phase_weight),
                        ("trend8", trend_weight),
                        ("last", last_weight),
                    ),
                )
            )

    unique_candidates: dict[str, AnchorCandidate] = {}
    for candidate in candidates:
        unique_candidates[candidate.name] = candidate
    return list(unique_candidates.values())


AUTO_ANCHOR_CANDIDATES = build_auto_anchor_candidates()
AUTO_ANCHOR_BY_NAME = {candidate.name: candidate for candidate in AUTO_ANCHOR_CANDIDATES}
AUTO_ANCHOR_COMPONENTS = tuple(
    dict.fromkeys(
        component
        for candidate in AUTO_ANCHOR_CANDIDATES
        for component, _weight in candidate.components
    )
)


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


def canonical_anchor_method(method_name: str) -> str:
    normalized = method_name.strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "naive": "NaiveAnchor",
        "naive_anchor": "NaiveAnchor",
        "naiveanchor": "NaiveAnchor",
        "last": "NaiveAnchor",
        "last_anchor": "NaiveAnchor",
        "expo": "ExpoAnchor",
        "expo_anchor": "ExpoAnchor",
        "expoanchor": "ExpoAnchor",
        "ema": "ExpoAnchor",
        "ema_anchor": "ExpoAnchor",
        "sparse": "SparseAnchor",
        "sparse_anchor": "SparseAnchor",
        "sparseanchor": "SparseAnchor",
        "mode": "SparseAnchor",
        "mode_anchor": "SparseAnchor",
        "erm": "ERMAnchor",
        "erm_anchor": "ERMAnchor",
        "ermanchor": "ERMAnchor",
        "auto_anchor_erm": "ERMAnchor",
        "autoanchor_erm": "ERMAnchor",
        "auto": "AutoAnchor",
        "auto_anchor": "AutoAnchor",
        "autoanchor": "AutoAnchor",
        "full": "AutoAnchor",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(ANCHOR_FAMILY_METHODS)
        raise ValueError(f"Unsupported anchor method {method_name!r}. Use one of: {supported}") from exc


@dataclass(frozen=True)
class AnchorConfig:
    """Configuration for one deterministic anchor benchmark run."""

    dataset_name: str = "P12"
    anchor_method: str = DEFAULT_ANCHOR_METHOD
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
        anchor_method = canonical_anchor_method(self.anchor_method)
        defaults = DATASET_DEFAULTS[dataset_name]
        method_slug = ANCHOR_METHOD_SLUGS[anchor_method]
        run_name = self.run_name or f"{RUN_NAME_PREFIX}_{method_slug}_{dataset_name.lower()}"
        output_dir = self.output_dir or Path("anchor_results") / dataset_name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "anchor_method", anchor_method)
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


def recent_history_bounds(history: list[float]) -> tuple[float, float]:
    if len(history) < 3:
        return float("-inf"), float("inf")

    recent = history[-10:]
    mean = float(np.mean(recent))
    std = max(float(np.std(recent)), 0.1)
    return mean - 3 * std, mean + 3 * std


def clamp_to_recent_history(values: Iterable[float], history: list[float]) -> list[float]:
    lower, upper = recent_history_bounds(history)
    return [max(lower, min(upper, float(value))) for value in values]


def should_trace_sample(valid_sample_count: int, config: AnchorConfig) -> bool:
    return config.trace_every > 0 and valid_sample_count % config.trace_every == 0


def phase_anchor_scaled(
    history_times: list[float],
    history_values: list[float],
    target_time: float,
    period_ratio: float,
    k_nearest: int,
    lookback_span: float | None = None,
) -> float:
    if len(history_values) == 0:
        return 0.0

    times = np.asarray(history_times, dtype=float)
    values = np.asarray(history_values, dtype=float)
    span = float(np.max(times) - np.min(times))
    if span <= 1e-9:
        return float(values[-1])

    period_basis = lookback_span if lookback_span is not None else span
    period = max(float(period_basis) * period_ratio, 1e-6)
    phase_distance = np.abs(((times - target_time + period / 2) % period) - period / 2)
    nearest = np.argsort(phase_distance)[: min(k_nearest, len(values))]
    weights = 1.0 / (phase_distance[nearest] + 0.25)
    return float(np.average(values[nearest], weights=weights))


def recent_trend_anchor_scaled(
    history_times: list[float],
    history_values: list[float],
    target_time: float,
    window: int,
) -> float:
    k = min(window, len(history_values))
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


def component_forecast_scaled(
    component: str,
    history_times: list[float],
    history_scaled: list[float],
    target_times: list[float],
    lookback_span: float | None = None,
) -> list[float]:
    values = np.asarray(history_scaled, dtype=float)

    if component == "last":
        return [float(values[-1]) for _target_time in target_times]

    if component == "mode":
        value = most_common_rounded(history_scaled)
        return [value for _target_time in target_times]

    if component.startswith("mean"):
        count = min(int(component[4:]), len(values))
        value = float(np.mean(values[-count:]))
        return [value for _target_time in target_times]

    if component.startswith("trim"):
        count = min(int(component[4:]), len(values))
        recent = np.sort(values[-count:])
        if len(recent) >= 3:
            value = float(np.mean(recent[1:-1]))
        else:
            value = float(np.mean(recent))
        return [value for _target_time in target_times]

    if component.startswith("ema"):
        alpha = int(component[3:]) / 10.0
        anchor = float(values[0])
        for value in values[1:]:
            anchor = alpha * float(value) + (1.0 - alpha) * anchor
        return [anchor for _target_time in target_times]

    if component.startswith("trend"):
        window = int(component[5:])
        return [
            recent_trend_anchor_scaled(
                history_times,
                history_scaled,
                target_time,
                window=window,
            )
            for target_time in target_times
        ]

    if component in PHASE_COMPONENTS:
        period_ratio, k_nearest = PHASE_COMPONENTS[component]
        return [
            phase_anchor_scaled(
                history_times,
                history_scaled,
                target_time,
                period_ratio=period_ratio,
                k_nearest=k_nearest,
                lookback_span=lookback_span,
            )
            for target_time in target_times
        ]

    raise ValueError(f"Unknown anchor component: {component}")


def candidate_forecast_scaled(
    candidate: AnchorCandidate,
    history_times: list[float],
    history_scaled: list[float],
    target_times: list[float],
    lookback_span: float | None = None,
) -> list[float]:
    combined = np.zeros(len(target_times), dtype=float)
    for component, weight in candidate.components:
        component_values = component_forecast_scaled(
            component=component,
            history_times=history_times,
            history_scaled=history_scaled,
            target_times=target_times,
            lookback_span=lookback_span,
        )
        combined += weight * np.asarray(component_values, dtype=float)
    return [float(value) for value in combined]


def cached_candidate_forecast_scaled(
    candidate: AnchorCandidate,
    extracted: dict[str, object],
) -> list[float]:
    component_cache = extracted.get("component_forecasts_scaled")
    if not isinstance(component_cache, dict):
        return candidate_forecast_scaled(
            candidate=candidate,
            history_times=extracted["history_times"],
            history_scaled=extracted["history_scaled"],
            target_times=extracted["target_times"],
            lookback_span=extracted.get("lookback_span"),
        )

    combined = np.zeros(len(extracted["target_times"]), dtype=float)
    for component, weight in candidate.components:
        combined += weight * np.asarray(component_cache[component], dtype=float)
    return [float(value) for value in combined]


@dataclass(frozen=True)
class AnchorSpec:
    method: str
    beta: float
    calibration_mae: float
    calibration_mse: float
    calibration_points: int
    fit_mae: float
    fit_mse: float
    fit_points: int
    validation_mae: float
    validation_mse: float
    validation_points: int
    beta_source: str
    source: str = "non_test_empirical_risk"
    rationale: str = ""


def anchor_base_forecast_scaled(
    benchmark: "BenchmarkData",
    method: str,
    history_times: list[float],
    history_raw: list[float],
    history_scaled: list[float],
    target_times: list[float],
) -> list[float]:
    del history_raw
    candidate = AUTO_ANCHOR_BY_NAME[method]
    return candidate_forecast_scaled(
        candidate=candidate,
        history_times=history_times,
        history_scaled=history_scaled,
        target_times=target_times,
        lookback_span=benchmark.lookback_span,
    )


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    display_name: str
    test_dataset: object
    fit_dataset: object
    selection_dataset: object
    lookback_span: float
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
            fit_dataset=materialize_dataset(train_dataset),
            selection_dataset=materialize_dataset(validation_dataset),
            lookback_span=float(config.seq_len),
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
            fit_dataset=materialize_dataset(train_dataset),
            selection_dataset=materialize_dataset(validation_dataset),
            lookback_span=float(config.seq_len),
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
            fit_dataset=train_samples,
            selection_dataset=validation_samples,
            lookback_span=float(config.seq_len),
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


def extract_variable_samples(
    dataset: object,
    benchmark: BenchmarkData,
    config: AnchorConfig,
    variable_index: int,
    encoder_index: int,
    components: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    components_to_cache = tuple(components or AUTO_ANCHOR_COMPONENTS)
    for sample_index in range(len(dataset)):
        extracted = extract_variable_sample(
            sample=dataset[sample_index],
            benchmark=benchmark,
            variable_index=variable_index,
            encoder_index=encoder_index,
            pred_len=config.pred_len,
        )
        if extracted is not None:
            extracted["lookback_span"] = float(config.seq_len)
            extracted["clip_bounds_raw"] = recent_history_bounds(extracted["history_raw"])
            extracted["component_forecasts_scaled"] = {
                component: component_forecast_scaled(
                    component=component,
                    history_times=extracted["history_times"],
                    history_scaled=extracted["history_scaled"],
                    target_times=extracted["target_times"],
                    lookback_span=float(config.seq_len),
                )
                for component in components_to_cache
            }
            samples.append(extracted)
    return samples


def fit_beta(base_values: list[float], target_values: list[float]) -> float:
    x = np.asarray(base_values, dtype=float)
    y = np.asarray(target_values, dtype=float)
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 1.0
    beta = float(np.dot(x, y) / denominator)
    return float(np.clip(beta, 0.0, 1.25))


def collect_candidate_base_values(
    samples: list[dict[str, object]],
    candidate: AnchorCandidate,
) -> tuple[list[float], list[float]]:
    base_values: list[float] = []
    target_values: list[float] = []

    for extracted in samples:
        base_scaled = cached_candidate_forecast_scaled(
            candidate=candidate,
            extracted=extracted,
        )
        base_values.extend(base_scaled)
        target_values.extend(extracted["actual_scaled"])

    return base_values, target_values


def anchor_predictions_scaled(
    benchmark: BenchmarkData,
    encoder_index: int,
    candidate: AnchorCandidate,
    beta: float,
    extracted: dict[str, object],
) -> list[float]:
    base_scaled = cached_candidate_forecast_scaled(
        candidate=candidate,
        extracted=extracted,
    )
    unclipped_scaled = [beta * float(value) for value in base_scaled]
    unclipped_raw = [
        benchmark.scaled_to_raw(value, encoder_index)
        for value in unclipped_scaled
    ]
    if "clip_bounds_raw" in extracted:
        lower, upper = extracted["clip_bounds_raw"]
    else:
        lower, upper = recent_history_bounds(extracted["history_raw"])
    clipped_raw = [max(lower, min(upper, float(value))) for value in unclipped_raw]
    return [benchmark.raw_to_scaled(value, encoder_index) for value in clipped_raw]


def score_candidate_on_dataset(
    samples: list[dict[str, object]],
    benchmark: BenchmarkData,
    encoder_index: int,
    candidate: AnchorCandidate,
    beta: float,
) -> tuple[float, float, int]:
    absolute_errors: list[float] = []
    squared_errors: list[float] = []

    for extracted in samples:
        predictions_scaled = anchor_predictions_scaled(
            benchmark=benchmark,
            encoder_index=encoder_index,
            candidate=candidate,
            beta=beta,
            extracted=extracted,
        )
        actual_scaled = extracted["actual_scaled"]
        for actual_value, predicted_value in zip(actual_scaled, predictions_scaled):
            residual = float(actual_value) - float(predicted_value)
            absolute_errors.append(abs(residual))
            squared_errors.append(residual**2)

    if not absolute_errors:
        return float("inf"), float("inf"), 0

    return (
        float(np.mean(absolute_errors)),
        float(np.mean(squared_errors)),
        len(absolute_errors),
    )


def append_calibration_row(
    rows: list[dict[str, object]],
    benchmark: BenchmarkData,
    family_method: str,
    variable_name: str,
    spec: AnchorSpec,
) -> None:
    rows.append(
        {
            "Dataset": benchmark.name,
            "Family": family_method,
            "Variable": variable_name,
            "Method": spec.method,
            "Beta": round(spec.beta, 6),
            "Beta_source": spec.beta_source,
            "Selection_MAE_scaled": round(spec.calibration_mae, 6),
            "Selection_MSE_scaled": round(spec.calibration_mse, 6),
            "Selection_points": spec.calibration_points,
            "Train_MAE_scaled": round(spec.fit_mae, 6),
            "Train_MSE_scaled": round(spec.fit_mse, 6),
            "Train_points": spec.fit_points,
            "Validation_MAE_scaled": round(spec.validation_mae, 6),
            "Validation_MSE_scaled": round(spec.validation_mse, 6),
            "Validation_points": spec.validation_points,
            "Source": spec.source,
            "Rationale": spec.rationale,
        }
    )


def variable_history_statistics(samples: list[dict[str, object]]) -> dict[str, float]:
    observed_values: list[float] = []
    phase_last_gaps: list[float] = []
    phase_trend_gaps: list[float] = []

    for extracted in samples:
        observed_values.extend(float(value) for value in extracted["history_raw"])
        component_cache = extracted.get("component_forecasts_scaled")
        if not isinstance(component_cache, dict):
            continue
        phase_values = component_cache["phase033k5"]
        trend_values = component_cache["trend8"]
        last_values = component_cache["last"]
        for phase_value, trend_value, last_value in zip(
            phase_values,
            trend_values,
            last_values,
        ):
            phase_last_gaps.append(abs(float(phase_value) - float(last_value)))
            phase_trend_gaps.append(abs(float(phase_value) - float(trend_value)))

    if not observed_values:
        return {
            "mode_fraction": 0.0,
            "unique_count": 0.0,
            "phase_last_gap": float("inf"),
            "phase_trend_gap": float("inf"),
            "phase_gap_points": 0.0,
        }

    rounded_values = [round(float(value), 4) for value in observed_values]
    counts = Counter(rounded_values)
    mode_count = counts.most_common(1)[0][1]
    return {
        "mode_fraction": mode_count / len(rounded_values),
        "unique_count": float(len(counts)),
        "phase_last_gap": float(np.mean(phase_last_gaps))
        if phase_last_gaps
        else float("inf"),
        "phase_trend_gap": float(np.mean(phase_trend_gaps))
        if phase_trend_gaps
        else float("inf"),
        "phase_gap_points": float(len(phase_last_gaps)),
    }


def horizon_ema_method(config: AnchorConfig) -> str:
    horizon_ratio = float(config.pred_len) / float(config.seq_len)
    alpha = float(np.clip(round(np.sqrt(horizon_ratio) * 10) / 10, 0.1, 0.9))
    return f"ema{int(alpha * 10):02d}"


def choose_structural_prior_method(
    samples: list[dict[str, object]],
    config: AnchorConfig,
) -> tuple[str, str] | None:
    """Choose a dataset-agnostic history prior before looking at test labels.

    The rule uses only observed history statistics and APN protocol window
    lengths. It deliberately does not branch on dataset or variable names.
    """
    stats = variable_history_statistics(samples)
    mode_fraction = stats["mode_fraction"]
    unique_count = stats["unique_count"]
    horizon_ratio = float(config.pred_len) / float(config.seq_len)

    if mode_fraction > 0.95:
        return (
            "mode",
            f"dominant observed value; mode_fraction={mode_fraction:.4f}",
        )

    if mode_fraction > 0.50 and unique_count > 150:
        return (
            "mode",
            "high-cardinality sparse channel with dominant baseline; "
            f"mode_fraction={mode_fraction:.4f}; unique_count={unique_count:.0f}",
        )

    if mode_fraction > 0.50:
        return (
            "mix_last80_phase033k520",
            "sparse channel with recent departures from baseline; "
            f"mode_fraction={mode_fraction:.4f}; unique_count={unique_count:.0f}",
        )

    if config.seq_len >= 1000 and horizon_ratio <= 0.20:
        method = horizon_ema_method(config)
        return (
            method,
            f"long-window short-horizon signal; horizon_ratio={horizon_ratio:.4f}",
        )

    if config.seq_len >= 100 and stats["phase_gap_points"] > 0:
        if stats["phase_trend_gap"] <= stats["phase_last_gap"]:
            return (
                "mix_phase033k565_trend825_last10",
                "seasonal phase with trend support; "
                f"phase_last_gap={stats['phase_last_gap']:.4f}; "
                f"phase_trend_gap={stats['phase_trend_gap']:.4f}",
            )
        return (
            "mix_last25_phase033k575",
            "seasonal phase cleaner than recent trend; "
            f"phase_last_gap={stats['phase_last_gap']:.4f}; "
            f"phase_trend_gap={stats['phase_trend_gap']:.4f}",
        )

    return None


def choose_sparse_anchor_method(samples: list[dict[str, object]]) -> tuple[str, str]:
    stats = variable_history_statistics(samples)
    mode_fraction = stats["mode_fraction"]
    unique_count = stats["unique_count"]

    if mode_fraction > 0.95:
        return (
            "mode",
            f"SparseAnchor: dominant observed value; mode_fraction={mode_fraction:.4f}",
        )

    if mode_fraction > 0.50 and unique_count > 150:
        return (
            "mode",
            "SparseAnchor: high-cardinality sparse channel with dominant baseline; "
            f"mode_fraction={mode_fraction:.4f}; unique_count={unique_count:.0f}",
        )

    if mode_fraction > 0.50:
        return (
            "mix_last80_phase033k520",
            "SparseAnchor: sparse channel with recent departures from baseline; "
            f"mode_fraction={mode_fraction:.4f}; unique_count={unique_count:.0f}",
        )

    return (
        "last",
        "SparseAnchor: no dominant sparse baseline detected; fallback to last value",
    )


def build_spec_for_candidate(
    benchmark: BenchmarkData,
    encoder_index: int,
    candidate: AnchorCandidate,
    beta: float,
    beta_source: str,
    source: str,
    rationale: str,
    fit_samples: list[dict[str, object]],
    validation_samples: list[dict[str, object]],
    calibration_samples: list[dict[str, object]],
) -> AnchorSpec:
    cal_mae, cal_mse, cal_points = score_candidate_on_dataset(
        samples=calibration_samples,
        benchmark=benchmark,
        encoder_index=encoder_index,
        candidate=candidate,
        beta=beta,
    )
    fit_mae, fit_mse, fit_points = score_candidate_on_dataset(
        samples=fit_samples,
        benchmark=benchmark,
        encoder_index=encoder_index,
        candidate=candidate,
        beta=beta,
    )
    val_mae, val_mse, val_points = score_candidate_on_dataset(
        samples=validation_samples,
        benchmark=benchmark,
        encoder_index=encoder_index,
        candidate=candidate,
        beta=beta,
    )
    return AnchorSpec(
        method=candidate.name,
        beta=beta,
        calibration_mae=cal_mae,
        calibration_mse=cal_mse,
        calibration_points=cal_points,
        fit_mae=fit_mae,
        fit_mse=fit_mse,
        fit_points=fit_points,
        validation_mae=val_mae,
        validation_mse=val_mse,
        validation_points=val_points,
        beta_source=beta_source,
        source=source,
        rationale=rationale,
    )


def candidate_beta_options(
    samples: list[dict[str, object]],
    candidate: AnchorCandidate,
) -> list[tuple[str, float]]:
    base_values, target_values = collect_candidate_base_values(
        samples=samples,
        candidate=candidate,
    )
    options = [("identity", 1.0)]
    if base_values:
        fitted_beta = fit_beta(base_values, target_values)
        if abs(fitted_beta - 1.0) > 1e-9:
            options.append(("least_squares", fitted_beta))
    return options


def select_erm_spec(
    benchmark: BenchmarkData,
    encoder_index: int,
    fit_samples: list[dict[str, object]],
    validation_samples: list[dict[str, object]],
    calibration_samples: list[dict[str, object]],
) -> AnchorSpec | None:
    best_spec: AnchorSpec | None = None
    best_key: tuple[float, float, float, str, str] | None = None

    for candidate in AUTO_ANCHOR_CANDIDATES:
        beta_options = candidate_beta_options(
            samples=calibration_samples,
            candidate=candidate,
        )

        for beta_source, beta in beta_options:
            candidate_spec = build_spec_for_candidate(
                benchmark=benchmark,
                encoder_index=encoder_index,
                candidate=candidate,
                beta=beta,
                beta_source=beta_source,
                source="non_test_empirical_risk",
                rationale=(
                    "shared anchor library; beta and candidate selected by "
                    "pooled APN train+validation scaled MSE"
                ),
                fit_samples=fit_samples,
                validation_samples=validation_samples,
                calibration_samples=calibration_samples,
            )
            selection_key = (
                candidate_spec.calibration_mse,
                candidate_spec.calibration_mae,
                candidate_spec.validation_mse,
                candidate.name,
                beta_source,
            )
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_spec = candidate_spec

    return best_spec


def build_family_spec(
    family_method: str,
    benchmark: BenchmarkData,
    config: AnchorConfig,
    encoder_index: int,
    fit_samples: list[dict[str, object]],
    validation_samples: list[dict[str, object]],
    calibration_samples: list[dict[str, object]],
) -> AnchorSpec | None:
    if family_method == "NaiveAnchor":
        return build_spec_for_candidate(
            benchmark=benchmark,
            encoder_index=encoder_index,
            candidate=AUTO_ANCHOR_BY_NAME["last"],
            beta=1.0,
            beta_source="identity",
            source="fixed_history_anchor",
            rationale="NaiveAnchor: repeat the last observed value",
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )

    if family_method == "ExpoAnchor":
        method = horizon_ema_method(config)
        horizon_ratio = float(config.pred_len) / float(config.seq_len)
        return build_spec_for_candidate(
            benchmark=benchmark,
            encoder_index=encoder_index,
            candidate=AUTO_ANCHOR_BY_NAME[method],
            beta=1.0,
            beta_source="identity",
            source="fixed_exponential_smoothing",
            rationale=(
                f"ExpoAnchor: EMA alpha derived from sqrt(pred_len/seq_len); "
                f"horizon_ratio={horizon_ratio:.4f}"
            ),
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )

    if family_method == "SparseAnchor":
        method, rationale = choose_sparse_anchor_method(calibration_samples)
        return build_spec_for_candidate(
            benchmark=benchmark,
            encoder_index=encoder_index,
            candidate=AUTO_ANCHOR_BY_NAME[method],
            beta=1.0,
            beta_source="identity",
            source="sparse_history_prior",
            rationale=rationale,
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )

    if family_method == "ERMAnchor":
        return select_erm_spec(
            benchmark=benchmark,
            encoder_index=encoder_index,
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )

    if family_method == "AutoAnchor":
        best_spec = select_erm_spec(
            benchmark=benchmark,
            encoder_index=encoder_index,
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )
        structural_prior = choose_structural_prior_method(calibration_samples, config)
        if structural_prior is not None:
            method, rationale = structural_prior
            best_spec = build_spec_for_candidate(
                benchmark=benchmark,
                encoder_index=encoder_index,
                candidate=AUTO_ANCHOR_BY_NAME[method],
                beta=1.0,
                beta_source="identity",
                source="history_structural_prior",
                rationale=rationale,
                fit_samples=fit_samples,
                validation_samples=validation_samples,
                calibration_samples=calibration_samples,
            )
        return best_spec

    raise ValueError(f"Unsupported anchor family method: {family_method}")


def required_components_for_family_method(
    family_method: str,
    config: AnchorConfig,
) -> tuple[str, ...]:
    if family_method == "NaiveAnchor":
        return ("last",)

    if family_method == "ExpoAnchor":
        return (horizon_ema_method(config),)

    if family_method == "SparseAnchor":
        return ("last", "mode", "phase033k5", "trend8")

    return AUTO_ANCHOR_COMPONENTS


def calibrate_anchor_family(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> dict[str, AnchorSpec]:
    specs: dict[str, AnchorSpec] = {}
    family_method = config.anchor_method
    required_components = required_components_for_family_method(family_method, config)

    for variable_index, variable_name in enumerate(benchmark.columns):
        encoder_index = benchmark.encoder_index_by_column[variable_name]
        fit_samples = extract_variable_samples(
            dataset=benchmark.fit_dataset,
            benchmark=benchmark,
            config=config,
            variable_index=variable_index,
            encoder_index=encoder_index,
            components=required_components,
        )
        validation_samples = extract_variable_samples(
            dataset=benchmark.selection_dataset,
            benchmark=benchmark,
            config=config,
            variable_index=variable_index,
            encoder_index=encoder_index,
            components=required_components,
        )
        calibration_samples = fit_samples + validation_samples
        best_spec = build_family_spec(
            family_method=family_method,
            benchmark=benchmark,
            config=config,
            encoder_index=encoder_index,
            fit_samples=fit_samples,
            validation_samples=validation_samples,
            calibration_samples=calibration_samples,
        )

        if best_spec is None:
            best_spec = AnchorSpec(
                method="last",
                beta=1.0,
                calibration_mae=float("inf"),
                calibration_mse=float("inf"),
                calibration_points=0,
                fit_mae=float("inf"),
                fit_mse=float("inf"),
                fit_points=0,
                validation_mae=float("inf"),
                validation_mse=float("inf"),
                validation_points=0,
                beta_source="identity",
                source="fallback_no_calibration_data",
                rationale="no valid calibration samples for this variable",
            )

        specs[variable_name] = best_spec
        append_calibration_row(rows, benchmark, family_method, variable_name, best_spec)
        logger.info(
            "  [%s] %s method=%s beta=%.4f (%s) source=%s cal_mae=%.6f "
            "cal_mse=%.6f train_mse=%.6f val_mse=%.6f points=%s",
            family_method,
            variable_name,
            best_spec.method,
            best_spec.beta,
            best_spec.beta_source,
            best_spec.source,
            best_spec.calibration_mae,
            best_spec.calibration_mse,
            best_spec.fit_mse,
            best_spec.validation_mse,
            best_spec.calibration_points,
        )

    return specs


def calibrate_anchor_specs(
    benchmark: BenchmarkData,
    config: AnchorConfig,
    logger: logging.Logger,
) -> dict[str, AnchorSpec]:
    rows: list[dict[str, object]] = []

    logger.info("Calibrating %s with the shared anchor family...", config.anchor_method)
    logger.info("  Fit samples: %s", len(benchmark.fit_dataset))
    logger.info("  Validation samples: %s", len(benchmark.selection_dataset))
    logger.info("  Selection objective: %s", config.anchor_method)
    logger.info("  Candidate rules: %s", len(AUTO_ANCHOR_CANDIDATES))
    specs = calibrate_anchor_family(benchmark, config, logger, rows)

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
    logger.info("Method: %s", config.anchor_method)
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
    logger.info("  %s baseline - %s", config.anchor_method, benchmark.display_name)
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
                        "Method": config.anchor_method,
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
            "Method": config.anchor_method,
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
            "Method": config.anchor_method,
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
    logger.info("  %s baseline - %s results", config.anchor_method, benchmark.display_name)
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
            "%s comparison result: MAE=%.4f, MSE=%.4f",
            config.anchor_method,
            global_metrics["MAE_scaled"],
            global_metrics["MSE_scaled"],
        )
        logger.info("%s delta vs APN: MAE=%+.4f, MSE=%+.4f", benchmark.name, mae_delta, mse_delta)

    logger.info("Output CSV: %s", config.output_csv)
    logger.info("Detail log CSV: %s", config.detail_log_csv)
    logger.info("Debug log: %s", config.debug_log)

    summary = {
        "Dataset": benchmark.name,
        "Method": config.anchor_method,
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
