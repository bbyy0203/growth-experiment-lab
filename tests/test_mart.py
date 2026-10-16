"""Tests for DuckDB marts, strict contracts, and data-quality reporting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import growth_experiment_lab.cli as cli_module
from growth_experiment_lab.data_validation import (
    MART_COLUMNS,
    DataValidationError,
    validate_mart,
    validate_raw_tables,
)
from growth_experiment_lab.mart import (
    MartBuildError,
    build_all_marts,
    build_mart_in_memory,
    pandas_reference_mart,
    validate_data_and_write_report,
)
from growth_experiment_lab.synthetic_data import generate_all, write_generation_result


@pytest.fixture(scope="module")
def mart_config(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from growth_experiment_lab.config import load_config

    config = load_config()
    root = tmp_path_factory.mktemp("mart-data")
    config["experiment"]["n_users"] = 600
    config["data_generation"]["output_dir"] = str(root / "data")
    config["mart"]["sql_dir"] = str(Path("sql").resolve())
    config["data_quality"]["report_path"] = str(root / "reports" / "quality.json")
    write_generation_result(generate_all(config), config["data_generation"]["output_dir"])
    return config


@pytest.fixture
def raw_frames(mart_config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset = generate_all(mart_config).datasets["randomized"]
    return {
        "users": dataset.users.copy(),
        "assignment": dataset.assignment.copy(),
        "outcomes": dataset.outcomes.copy(),
    }


def test_sql_is_explicit() -> None:
    for path in Path("sql").glob("build_*_mart.sql"):
        normalized = " ".join(path.read_text(encoding="utf-8").lower().split())
        assert "select *" not in normalized


@pytest.mark.parametrize("dataset_type", ["randomized", "observational"])
def test_sql_mart_matches_independent_pandas_merge(
    mart_config: dict[str, Any], dataset_type: str
) -> None:
    result = build_mart_in_memory(mart_config, dataset_type)  # type: ignore[arg-type]
    tables = {
        name: pd.read_parquet(path) for name, path in result.input_paths.items()
    }
    expected = pandas_reference_mart(tables)

    assert len(result.mart) == len(tables["users"])
    assert result.mart["user_id"].is_unique
    assert list(result.mart.columns) == MART_COLUMNS
    assert result.crosscheck_passed is True
    pd.testing.assert_frame_equal(
        result.mart, expected, check_dtype=False, check_categorical=False
    )


def test_mart_excludes_generation_truth(mart_config: dict[str, Any]) -> None:
    forbidden = {
        "assignment_probability",
        "p0",
        "p1",
        "true_effect",
        "true_ate",
        "potential_outcome",
    }
    result = build_mart_in_memory(mart_config, "observational")

    assert forbidden.isdisjoint(result.mart.columns)


def test_dataset_types_cannot_be_mixed(raw_frames: dict[str, pd.DataFrame]) -> None:
    raw_frames["assignment"]["dataset_type"] = "observational"

    with pytest.raises(DataValidationError, match="dataset_type"):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


@pytest.mark.parametrize("table_name", ["users", "assignment", "outcomes"])
def test_missing_user_fails(
    raw_frames: dict[str, pd.DataFrame], table_name: str
) -> None:
    raw_frames[table_name] = raw_frames[table_name].iloc[:-1].copy()

    with pytest.raises(DataValidationError, match="row count"):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


@pytest.mark.parametrize("table_name", ["assignment", "outcomes"])
def test_duplicate_user_fails(
    raw_frames: dict[str, pd.DataFrame], table_name: str
) -> None:
    raw_frames[table_name] = pd.concat(
        [raw_frames[table_name], raw_frames[table_name].iloc[[0]]], ignore_index=True
    )

    with pytest.raises(DataValidationError, match="duplicate user_id"):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


def test_cross_group_duplicate_fails(raw_frames: dict[str, pd.DataFrame]) -> None:
    duplicate = raw_frames["assignment"].iloc[[0]].copy()
    duplicate["treatment"] = 1 - duplicate["treatment"]
    raw_frames["assignment"] = pd.concat(
        [raw_frames["assignment"], duplicate], ignore_index=True
    )

    with pytest.raises(DataValidationError, match="cross-group"):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


@pytest.mark.parametrize(
    ("table_name", "column", "value", "message"),
    [
        ("assignment", "treatment", 2, "treatment"),
        ("outcomes", "clicked", 3, "clicked"),
        ("outcomes", "next_day_retention", -1, "next_day_retention"),
        ("outcomes", "negative_feedback", 4, "negative_feedback"),
        ("outcomes", "watch_time_minutes", -0.1, "watch_time_minutes"),
        ("users", "pre_7d_sessions", -1, "pre_7d_sessions"),
        ("users", "pre_7d_watch_time_minutes", -1.0, "pre_7d_watch_time_minutes"),
        ("users", "pre_7d_click_days", 8, "pre_7d_click_days"),
        ("users", "is_new_user", 2, "is_new_user"),
        ("users", "assignment_date", "not-a-date", "assignment_date"),
    ],
)
def test_invalid_values_fail(
    raw_frames: dict[str, pd.DataFrame],
    table_name: str,
    column: str,
    value: object,
    message: str,
) -> None:
    if column == "assignment_date":
        raw_frames[table_name][column] = raw_frames[table_name][column].astype(object)
    raw_frames[table_name].loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


def test_missing_required_column_fails(raw_frames: dict[str, pd.DataFrame]) -> None:
    raw_frames["users"] = raw_frames["users"].drop(columns=["region"])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_raw_tables(**raw_frames, dataset_type="randomized", expected_n_users=600)


@pytest.mark.parametrize(
    "column", ["assignment_probability", "p0", "p1", "true_effect", "true_ate", "ground_truth_ate"]
)
def test_leakage_field_in_mart_fails(
    mart_config: dict[str, Any], column: str
) -> None:
    mart = build_mart_in_memory(mart_config, "randomized").mart
    mart[column] = 0.5

    with pytest.raises(DataValidationError, match="forbidden leakage"):
        validate_mart(mart, "randomized", 600)


def test_build_refuses_overwrite_and_allows_flag(mart_config: dict[str, Any]) -> None:
    results = build_all_marts(mart_config)
    assert all(result.output_path.is_file() for result in results)

    with pytest.raises(MartBuildError, match="--overwrite"):
        build_all_marts(mart_config)

    overwritten = build_all_marts(mart_config, overwrite=True)
    assert all(result.output_path.is_file() for result in overwritten)


def test_validate_data_writes_complete_json(mart_config: dict[str, Any]) -> None:
    build_all_marts(mart_config, overwrite=True)
    report = validate_data_and_write_report(mart_config)
    report_path = Path(mart_config["data_quality"]["report_path"])
    written = json.loads(report_path.read_text(encoding="utf-8"))

    assert report == written
    assert set(report) == {"synthetic_data_only", "checks_passed", "failed_checks", "datasets"}
    assert set(report["datasets"]) == {"randomized", "observational"}
    for summary in report["datasets"].values():
        assert {
            "dataset_type",
            "checks_passed",
            "failed_checks",
            "total_users",
            "treatment_users",
            "control_users",
            "treatment_share",
            "missing_values",
            "duplicate_user_ids",
            "binary_metric_means",
            "watch_time_minutes",
            "pretreatment_numeric",
            "pretreatment_categorical_counts",
            "sql_pandas_crosscheck",
        } == set(summary)


def test_validate_data_cli_success(
    mart_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: deepcopy(mart_config))

    assert cli_module.main(["validate-data", "--config", "unused.toml"]) == 0


def test_validate_data_cli_failure(
    mart_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid = deepcopy(mart_config)
    invalid["data_generation"]["output_dir"] = str(tmp_path / "missing")
    monkeypatch.setattr(cli_module, "load_config", lambda _: invalid)

    with pytest.raises(SystemExit) as error:
        cli_module.main(["validate-data", "--config", "unused.toml"])
    assert error.value.code != 0
    failure_report = json.loads(
        Path(invalid["data_quality"]["report_path"]).read_text(encoding="utf-8")
    )
    assert failure_report["checks_passed"] is False
    assert failure_report["failed_checks"]
