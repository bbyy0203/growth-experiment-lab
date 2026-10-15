"""Reproducible synthetic user, assignment, and outcome data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

DatasetType = Literal["randomized", "observational"]
DATASET_TYPES: tuple[DatasetType, ...] = ("randomized", "observational")
USER_COLUMNS = [
    "user_id",
    "assignment_date",
    "device_type",
    "region",
    "is_new_user",
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
]
ASSIGNMENT_COLUMNS = ["user_id", "treatment", "assignment_probability", "dataset_type"]
OUTCOME_COLUMNS = [
    "user_id",
    "clicked",
    "watch_time_minutes",
    "next_day_retention",
    "negative_feedback",
]
PROPENSITY_INPUT_COLUMNS = [
    "device_type",
    "region",
    "is_new_user",
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
]


class DataGenerationError(RuntimeError):
    """Raised when synthetic outputs cannot be generated safely."""


@dataclass(frozen=True)
class GeneratedDataset:
    """The three public tables and internal truth summary for one dataset type."""

    users: pd.DataFrame
    assignment: pd.DataFrame
    outcomes: pd.DataFrame
    true_effects: dict[str, float]
    probability_clipping: dict[str, bool]


@dataclass(frozen=True)
class GenerationResult:
    """Both synthetic datasets and their shared metadata."""

    datasets: dict[DatasetType, GeneratedDataset]
    metadata: dict[str, Any]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _standardize(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    standard_deviation = array.std()
    if standard_deviation == 0:
        return np.zeros_like(array)
    return (array - array.mean()) / standard_deviation


def _generate_users(
    n_users: int,
    dataset_type: DatasetType,
    assignment_start_date: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    is_new_user = rng.binomial(1, 0.32, n_users).astype("int8")
    latent_activity = rng.normal(0.0, 1.0, n_users) - 0.55 * is_new_user
    pre_7d_sessions = rng.poisson(np.exp(1.15 + 0.38 * latent_activity)).astype("int16")
    pre_7d_watch = np.where(
        pre_7d_sessions == 0,
        0.0,
        rng.gamma(shape=2.2, scale=7.0, size=n_users) * np.maximum(pre_7d_sessions, 1),
    )
    click_probability = np.clip(_sigmoid(-1.0 + 0.35 * latent_activity), 0.03, 0.85)
    pre_7d_click_days = rng.binomial(7, click_probability).astype("int8")
    activity_score = (
        0.45 * np.log1p(pre_7d_sessions)
        + 0.35 * np.log1p(pre_7d_watch)
        + 0.20 * pre_7d_click_days
    )
    activity_score = (activity_score - activity_score.mean()) / activity_score.std()
    assignment_offsets = rng.integers(0, 14, size=n_users)

    return pd.DataFrame(
        {
            "user_id": [f"{dataset_type}_{index:07d}" for index in range(1, n_users + 1)],
            "assignment_date": pd.Timestamp(assignment_start_date)
            + pd.to_timedelta(assignment_offsets, unit="D"),
            "device_type": pd.Categorical(
                rng.choice(["android", "ios", "web"], n_users, p=[0.52, 0.34, 0.14])
            ),
            "region": pd.Categorical(
                rng.choice(["east", "north", "south", "west"], n_users, p=[0.31, 0.24, 0.28, 0.17])
            ),
            "is_new_user": is_new_user,
            "pre_7d_sessions": pre_7d_sessions,
            "pre_7d_watch_time_minutes": np.round(pre_7d_watch, 3),
            "pre_7d_click_days": pre_7d_click_days,
            "pre_experiment_activity_score": np.round(activity_score, 6),
        },
        columns=USER_COLUMNS,
    )


def assignment_probabilities(
    users: pd.DataFrame,
    dataset_type: DatasetType,
    treatment_share: float,
    propensity_min: float,
    propensity_max: float,
) -> np.ndarray:
    """Return assignment probabilities using only documented pre-treatment columns."""
    if dataset_type == "randomized":
        return np.full(len(users), treatment_share, dtype=float)

    activity = _standardize(users["pre_experiment_activity_score"])
    sessions = _standardize(users["pre_7d_sessions"])
    click_days = _standardize(users["pre_7d_click_days"])
    linear_predictor = (
        -0.08
        + 0.52 * activity
        + 0.22 * sessions
        + 0.20 * click_days
        - 0.32 * users["is_new_user"].to_numpy()
        + 0.18 * (users["device_type"].astype(str).to_numpy() == "ios")
        + 0.12 * (users["region"].astype(str).to_numpy() == "east")
    )
    return np.clip(_sigmoid(linear_predictor), propensity_min, propensity_max)


def _binary_potential_probabilities(
    users: pd.DataFrame,
    baseline: float,
    effect: float,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    activity = _standardize(users["pre_experiment_activity_score"])
    sessions = _standardize(users["pre_7d_sessions"])
    is_new = users["is_new_user"].to_numpy(dtype=float)
    ios = (users["device_type"].astype(str).to_numpy() == "ios").astype(float)
    region_east = (users["region"].astype(str).to_numpy() == "east").astype(float)

    coefficients = {
        "next_day_retention": (0.060, 0.025, -0.045, 0.012, 0.012),
        "clicked": (0.055, 0.030, -0.025, 0.018, 0.010),
        "negative_feedback": (-0.020, -0.008, 0.028, -0.004, 0.006),
    }
    activity_coef, sessions_coef, new_coef, ios_coef, region_coef = coefficients[metric]
    raw_p0 = (
        baseline
        + activity_coef * activity
        + sessions_coef * sessions
        + new_coef * (is_new - is_new.mean())
        + ios_coef * (ios - ios.mean())
        + region_coef * (region_east - region_east.mean())
    )
    raw_p1 = raw_p0 + effect
    p0 = np.clip(raw_p0, 0.001, 0.999)
    p1 = np.clip(raw_p1, 0.001, 0.999)
    clipped = bool(np.any(raw_p0 != p0) or np.any(raw_p1 != p1))
    return p0, p1, clipped


def _watch_time_expectations(
    users: pd.DataFrame, baseline_mean: float, effect: float
) -> tuple[np.ndarray, np.ndarray]:
    activity = _standardize(users["pre_experiment_activity_score"])
    prior_watch = _standardize(users["pre_7d_watch_time_minutes"])
    is_new = users["is_new_user"].to_numpy(dtype=float)
    raw_mu0 = baseline_mean + 3.2 * activity + 1.4 * prior_watch - 2.0 * (is_new - is_new.mean())
    mu0 = np.clip(raw_mu0, 1.0, None)
    mu1 = mu0 + effect
    return mu0, mu1


def _generate_outcomes(
    users: pd.DataFrame,
    treatment: np.ndarray,
    metrics: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, bool]]:
    binary_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    true_effects: dict[str, float] = {}
    clipping: dict[str, bool] = {}
    for metric in ("clicked", "next_day_retention", "negative_feedback"):
        specification = metrics[metric]
        p0, p1, was_clipped = _binary_potential_probabilities(
            users,
            float(specification["control_baseline"]),
            float(specification["treatment_effect"]),
            metric,
        )
        binary_probabilities[metric] = (p0, p1)
        true_effects[metric] = float(np.mean(p1 - p0))
        clipping[metric] = was_clipped

    watch_specification = metrics["watch_time_minutes"]
    watch_mu0, watch_mu1 = _watch_time_expectations(
        users,
        float(watch_specification["control_mean"]),
        float(watch_specification["treatment_effect"]),
    )
    true_effects["watch_time_minutes"] = float(np.mean(watch_mu1 - watch_mu0))
    clipping["watch_time_minutes"] = False

    observed_binary: dict[str, np.ndarray] = {}
    for metric, (p0, p1) in binary_probabilities.items():
        probability = np.where(treatment == 1, p1, p0)
        observed_binary[metric] = rng.binomial(1, probability).astype("int8")

    activity = _standardize(users["pre_experiment_activity_score"])
    positive_probability0 = np.clip(_sigmoid(0.45 + 0.38 * activity), 0.25, 0.92)
    positive_probability1 = np.clip(positive_probability0 + 0.025, 0.25, 0.95)
    positive_probability = np.where(treatment == 1, positive_probability1, positive_probability0)
    expected_watch = np.where(treatment == 1, watch_mu1, watch_mu0)
    positive_mean = expected_watch / positive_probability
    watched = rng.binomial(1, positive_probability)
    watch_time = watched * rng.gamma(shape=2.2, scale=positive_mean / 2.2)

    outcomes = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy(),
            "clicked": observed_binary["clicked"],
            "watch_time_minutes": np.round(watch_time, 3),
            "next_day_retention": observed_binary["next_day_retention"],
            "negative_feedback": observed_binary["negative_feedback"],
        },
        columns=OUTCOME_COLUMNS,
    )
    return outcomes, true_effects, clipping


def generate_dataset(
    config: dict[str, Any], dataset_type: DatasetType, rng: np.random.Generator
) -> GeneratedDataset:
    """Generate one complete synthetic dataset without writing files."""
    experiment = config["experiment"]
    generation = config["data_generation"]
    propensity = config["observational"]["generation"]
    users_rng, assignment_rng, outcomes_rng = [np.random.default_rng(seed) for seed in rng.spawn(3)]
    users = _generate_users(
        int(experiment["n_users"]),
        dataset_type,
        str(generation["assignment_start_date"]),
        users_rng,
    )
    probabilities = assignment_probabilities(
        users,
        dataset_type,
        float(experiment["treatment_share"]),
        float(propensity["propensity_min"]),
        float(propensity["propensity_max"]),
    )
    treatment = assignment_rng.binomial(1, probabilities).astype("int8")
    assignment = pd.DataFrame(
        {
            "user_id": users["user_id"].to_numpy(),
            "treatment": treatment,
            "assignment_probability": probabilities,
            "dataset_type": dataset_type,
        },
        columns=ASSIGNMENT_COLUMNS,
    )
    outcomes, true_effects, clipping = _generate_outcomes(
        users, treatment, config["metrics"], outcomes_rng
    )
    return GeneratedDataset(users, assignment, outcomes, true_effects, clipping)


def generate_all(config: dict[str, Any]) -> GenerationResult:
    """Generate randomized and observational datasets deterministically."""
    seed = int(config["project"]["random_seed"])
    root_rng = np.random.default_rng(seed)
    dataset_rngs = [np.random.default_rng(child) for child in root_rng.spawn(2)]
    datasets = {
        dataset_type: generate_dataset(config, dataset_type, dataset_rng)
        for dataset_type, dataset_rng in zip(DATASET_TYPES, dataset_rngs, strict=True)
    }
    nominal_parameters = {
        metric: {
            key: value
            for key, value in specification.items()
            if key in {"control_baseline", "control_mean", "treatment_effect"}
        }
        for metric, specification in config["metrics"].items()
    }
    metadata: dict[str, Any] = {
        "random_seed": seed,
        "n_users": int(config["experiment"]["n_users"]),
        "data_generation_version": config["data_generation"]["version"],
        "true_sample_ate": {
            dataset_type: dataset.true_effects for dataset_type, dataset in datasets.items()
        },
        "nominal_generation_parameters": nominal_parameters,
        "probability_clipping": {
            dataset_type: dataset.probability_clipping
            for dataset_type, dataset in datasets.items()
        },
        "effects_recomputed_after_clipping": True,
        "actual_treatment_share": {
            dataset_type: float(dataset.assignment["treatment"].mean())
            for dataset_type, dataset in datasets.items()
        },
        "config_summary": {
            "treatment_share": float(config["experiment"]["treatment_share"]),
            "experimental_unit": config["experiment"]["unit"],
            "estimand": config["experiment"]["estimand"],
            "primary_metric": "next_day_retention",
            "predefined_segment": config["experiment"]["segment"],
        },
    }
    return GenerationResult(datasets, metadata)


def output_paths(output_root: Path) -> list[Path]:
    """Return every file produced by the stage-2A generator."""
    paths = []
    for dataset_type in DATASET_TYPES:
        paths.extend(
            output_root / "raw" / dataset_type / filename
            for filename in ("users.parquet", "assignment.parquet", "outcomes.parquet")
        )
    paths.append(output_root / "metadata" / "true_effects.json")
    return paths


def write_generation_result(
    result: GenerationResult, output_root: str | Path, overwrite: bool = False
) -> list[Path]:
    """Write Parquet tables and JSON metadata, refusing accidental overwrite."""
    root = Path(output_root)
    paths = output_paths(root)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise DataGenerationError(
            f"Output files already exist: {formatted}. Use --overwrite to replace them."
        )

    for dataset_type, dataset in result.datasets.items():
        directory = root / "raw" / dataset_type
        directory.mkdir(parents=True, exist_ok=True)
        dataset.users.to_parquet(directory / "users.parquet", index=False)
        dataset.assignment.to_parquet(directory / "assignment.parquet", index=False)
        dataset.outcomes.to_parquet(directory / "outcomes.parquet", index=False)
    metadata_path = root / "metadata" / "true_effects.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths


def generate_and_write(
    config: dict[str, Any], overwrite: bool = False
) -> tuple[GenerationResult, list[Path]]:
    """Generate both datasets and write them under the configured output directory."""
    output_root = Path(config["data_generation"]["output_dir"])
    existing = [path for path in output_paths(output_root) if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise DataGenerationError(
            f"Output files already exist: {formatted}. Use --overwrite to replace them."
        )
    result = generate_all(config)
    return result, write_generation_result(result, output_root, overwrite=overwrite)
