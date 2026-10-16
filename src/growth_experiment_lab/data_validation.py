"""Strict data contracts and descriptive quality summaries for synthetic data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from growth_experiment_lab.synthetic_data import (
    ASSIGNMENT_COLUMNS,
    OUTCOME_COLUMNS,
    USER_COLUMNS,
    DatasetType,
)

MART_COLUMNS = [
    "user_id",
    "assignment_date",
    "dataset_type",
    "treatment",
    "device_type",
    "region",
    "is_new_user",
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
    "clicked",
    "watch_time_minutes",
    "next_day_retention",
    "negative_feedback",
]
PRETREATMENT_COVARIATES = [
    "device_type",
    "region",
    "is_new_user",
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
]
BINARY_OUTCOMES = ["clicked", "next_day_retention", "negative_feedback"]
FORBIDDEN_EXACT_COLUMNS = {
    "assignment_probability",
    "p0",
    "p1",
    "true_effect",
    "true_ate",
    "potential_outcome",
}
FORBIDDEN_COLUMN_MARKERS = (
    "true_effect",
    "true_ate",
    "potential_outcome",
    "simulation_truth",
    "ground_truth",
    "latent_outcome",
)


class DataValidationError(ValueError):
    """Raised when raw data or a formal analysis mart violates its contract."""


def _require_exact_columns(frame: pd.DataFrame, expected: list[str], table_name: str) -> None:
    actual = set(frame.columns)
    expected_set = set(expected)
    missing = expected_set - actual
    extra = actual - expected_set
    if missing:
        raise DataValidationError(
            f"{table_name} is missing required columns: {', '.join(sorted(missing))}"
        )
    if extra:
        raise DataValidationError(
            f"{table_name} contains undeclared columns: {', '.join(sorted(extra))}"
        )


def _require_no_missing(frame: pd.DataFrame, table_name: str) -> None:
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        details = ", ".join(f"{column}={int(count)}" for column, count in missing.items())
        raise DataValidationError(f"{table_name} contains missing values: {details}")


def _require_unique_users(frame: pd.DataFrame, table_name: str) -> None:
    duplicate_count = int(frame["user_id"].duplicated().sum())
    if duplicate_count:
        if table_name == "assignment":
            conflicts = frame.groupby("user_id")["treatment"].nunique()
            if bool((conflicts > 1).any()):
                raise DataValidationError("assignment contains cross-group duplicate users")
        raise DataValidationError(
            f"{table_name} contains {duplicate_count} duplicate user_id rows"
        )


def _require_binary(frame: pd.DataFrame, column: str, table_name: str) -> None:
    invalid = ~frame[column].isin([0, 1])
    if bool(invalid.any()):
        values = sorted(frame.loc[invalid, column].astype(str).unique())
        raise DataValidationError(f"{table_name}.{column} must contain only 0/1; found {values}")


def _reject_leakage_columns(columns: pd.Index) -> None:
    normalized = {str(column).lower() for column in columns}
    exact = normalized.intersection(FORBIDDEN_EXACT_COLUMNS)
    marked = {
        column
        for column in normalized
        if any(marker in column for marker in FORBIDDEN_COLUMN_MARKERS)
    }
    forbidden = exact.union(marked)
    if forbidden:
        raise DataValidationError(
            "Formal analysis mart contains forbidden leakage fields: "
            f"{', '.join(sorted(forbidden))}"
        )


def _require_valid_dates(frame: pd.DataFrame, column: str, table_name: str) -> None:
    try:
        parsed_dates = pd.to_datetime(frame[column], errors="raise", format="mixed")
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{table_name}.{column} must contain valid dates") from error
    if bool(parsed_dates.isna().any()):
        raise DataValidationError(f"{table_name}.{column} must contain valid dates")


def validate_raw_tables(
    users: pd.DataFrame,
    assignment: pd.DataFrame,
    outcomes: pd.DataFrame,
    dataset_type: DatasetType,
    expected_n_users: int,
) -> None:
    """Validate raw table schemas, values, cardinality, and one-to-one keys."""
    frames = {"users": users, "assignment": assignment, "outcomes": outcomes}
    expected_columns = {
        "users": USER_COLUMNS,
        "assignment": ASSIGNMENT_COLUMNS,
        "outcomes": OUTCOME_COLUMNS,
    }
    for table_name, frame in frames.items():
        _require_exact_columns(frame, expected_columns[table_name], table_name)
        _require_no_missing(frame, table_name)
        _require_unique_users(frame, table_name)
        if len(frame) != expected_n_users:
            raise DataValidationError(
                f"{table_name} row count {len(frame)} does not match expected {expected_n_users}"
            )

    user_ids = set(users["user_id"])
    for table_name, frame in (("assignment", assignment), ("outcomes", outcomes)):
        frame_ids = set(frame["user_id"])
        missing = user_ids - frame_ids
        orphan = frame_ids - user_ids
        if missing or orphan:
            raise DataValidationError(
                f"{table_name} user_id set differs from users: "
                f"missing={len(missing)}, orphan={len(orphan)}"
            )

    observed_types = set(assignment["dataset_type"].astype(str).unique())
    if observed_types != {dataset_type}:
        raise DataValidationError(
            f"assignment dataset_type must be only '{dataset_type}', found {sorted(observed_types)}"
        )
    _require_binary(assignment, "treatment", "assignment")
    for column in BINARY_OUTCOMES:
        _require_binary(outcomes, column, "outcomes")
    if bool((outcomes["watch_time_minutes"] < 0).any()):
        raise DataValidationError("outcomes.watch_time_minutes must be non-negative")
    if bool((users["pre_7d_sessions"] < 0).any()):
        raise DataValidationError("users.pre_7d_sessions must be non-negative")
    if bool((users["pre_7d_watch_time_minutes"] < 0).any()):
        raise DataValidationError("users.pre_7d_watch_time_minutes must be non-negative")
    if not bool(users["pre_7d_click_days"].between(0, 7).all()):
        raise DataValidationError("users.pre_7d_click_days must be between 0 and 7")
    _require_binary(users, "is_new_user", "users")
    _require_valid_dates(users, "assignment_date", "users")


def validate_mart(mart: pd.DataFrame, dataset_type: DatasetType, expected_n_users: int) -> None:
    """Validate the strict whitelist and values of a formal user-level analysis mart."""
    _reject_leakage_columns(mart.columns)
    _require_exact_columns(mart, MART_COLUMNS, f"{dataset_type} mart")
    _require_no_missing(mart, f"{dataset_type} mart")
    _require_unique_users(mart, f"{dataset_type} mart")
    if len(mart) != expected_n_users:
        raise DataValidationError(
            f"{dataset_type} mart row count {len(mart)} does not match expected {expected_n_users}"
        )
    observed_types = set(mart["dataset_type"].astype(str).unique())
    if observed_types != {dataset_type}:
        raise DataValidationError(
            f"{dataset_type} mart contains wrong dataset_type values: {sorted(observed_types)}"
        )
    _require_binary(mart, "treatment", f"{dataset_type} mart")
    _require_binary(mart, "is_new_user", f"{dataset_type} mart")
    for column in BINARY_OUTCOMES:
        _require_binary(mart, column, f"{dataset_type} mart")
    if bool((mart["watch_time_minutes"] < 0).any()):
        raise DataValidationError(f"{dataset_type} mart watch_time_minutes must be non-negative")
    if bool((mart["pre_7d_sessions"] < 0).any()):
        raise DataValidationError(f"{dataset_type} mart pre_7d_sessions must be non-negative")
    if bool((mart["pre_7d_watch_time_minutes"] < 0).any()):
        raise DataValidationError(
            f"{dataset_type} mart pre_7d_watch_time_minutes must be non-negative"
        )
    if not bool(mart["pre_7d_click_days"].between(0, 7).all()):
        raise DataValidationError(f"{dataset_type} mart pre_7d_click_days must be between 0 and 7")
    _require_valid_dates(mart, "assignment_date", f"{dataset_type} mart")


def quality_summary(mart: pd.DataFrame, dataset_type: DatasetType) -> dict[str, Any]:
    """Return descriptive, non-causal data-quality statistics for one valid mart."""
    treatment_count = int(mart["treatment"].sum())
    total_users = len(mart)
    numeric_covariates = [
        "pre_7d_sessions",
        "pre_7d_watch_time_minutes",
        "pre_7d_click_days",
        "pre_experiment_activity_score",
    ]
    numeric_summary = {
        column: {
            "min": float(mart[column].min()),
            "mean": float(mart[column].mean()),
            "median": float(mart[column].median()),
            "max": float(mart[column].max()),
        }
        for column in numeric_covariates
    }
    categorical_summary = {
        column: {
            str(value): int(count)
            for value, count in mart[column].value_counts(dropna=False).sort_index().items()
        }
        for column in ("device_type", "region", "is_new_user")
    }
    return {
        "dataset_type": dataset_type,
        "checks_passed": True,
        "failed_checks": [],
        "total_users": total_users,
        "treatment_users": treatment_count,
        "control_users": total_users - treatment_count,
        "treatment_share": float(mart["treatment"].mean()),
        "missing_values": int(mart.isna().sum().sum()),
        "duplicate_user_ids": int(mart["user_id"].duplicated().sum()),
        "binary_metric_means": {
            column: float(mart[column].mean()) for column in BINARY_OUTCOMES
        },
        "watch_time_minutes": {
            "min": float(mart["watch_time_minutes"].min()),
            "mean": float(mart["watch_time_minutes"].mean()),
            "median": float(mart["watch_time_minutes"].median()),
            "max": float(mart["watch_time_minutes"].max()),
            "zero_share": float(mart["watch_time_minutes"].eq(0).mean()),
        },
        "pretreatment_numeric": numeric_summary,
        "pretreatment_categorical_counts": categorical_summary,
    }
