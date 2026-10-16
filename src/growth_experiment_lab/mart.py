"""Build and cross-check user-level analysis marts with DuckDB SQL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from growth_experiment_lab.data_validation import (
    MART_COLUMNS,
    DataValidationError,
    quality_summary,
    validate_mart,
    validate_raw_tables,
)
from growth_experiment_lab.synthetic_data import DATASET_TYPES, DatasetType


class MartBuildError(RuntimeError):
    """Raised when a mart cannot be built without violating its contract."""


@dataclass(frozen=True)
class MartBuildResult:
    """A validated mart and its build provenance."""

    dataset_type: DatasetType
    mart: pd.DataFrame
    input_paths: dict[str, Path]
    output_path: Path
    crosscheck_passed: bool


def raw_paths(data_root: Path, dataset_type: DatasetType) -> dict[str, Path]:
    """Return the three required raw-table paths for one dataset type."""
    base = data_root / "raw" / dataset_type
    return {table: base / f"{table}.parquet" for table in ("users", "assignment", "outcomes")}


def mart_path(data_root: Path, dataset_type: DatasetType) -> Path:
    """Return the processed mart path for one dataset type."""
    return data_root / "processed" / dataset_type / "experiment_mart.parquet"


def _read_raw_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise MartBuildError(f"Required raw Parquet files are missing: {', '.join(missing)}")
    return {table: pd.read_parquet(path) for table, path in paths.items()}


def pandas_reference_mart(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Independently join raw tables with pandas using one-to-one validation."""
    users = tables["users"]
    assignment = tables["assignment"].drop(columns=["assignment_probability"])
    outcomes = tables["outcomes"]
    merged = users.merge(assignment, on="user_id", how="inner", validate="one_to_one")
    merged = merged.merge(outcomes, on="user_id", how="inner", validate="one_to_one")
    return merged[MART_COLUMNS].sort_values("user_id").reset_index(drop=True)


def crosscheck_with_pandas(sql_mart: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """Require the SQL result to equal an independent pandas merge."""
    expected = pandas_reference_mart(tables)
    actual = sql_mart[MART_COLUMNS].sort_values("user_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            check_categorical=False,
        )
    except AssertionError as error:
        raise MartBuildError(f"DuckDB SQL mart differs from pandas reference: {error}") from error


def build_mart_in_memory(
    config: dict[str, Any], dataset_type: DatasetType
) -> MartBuildResult:
    """Validate raw data, execute explicit SQL, validate and cross-check the result."""
    data_root = Path(config["data_generation"]["output_dir"])
    paths = raw_paths(data_root, dataset_type)
    tables = _read_raw_tables(paths)
    expected_n_users = int(config["experiment"]["n_users"])
    validate_raw_tables(
        tables["users"], tables["assignment"], tables["outcomes"], dataset_type, expected_n_users
    )

    sql_path = Path(config["mart"]["sql_dir"]) / f"build_{dataset_type}_mart.sql"
    if not sql_path.is_file():
        raise MartBuildError(f"Mart SQL file not found: {sql_path}")
    sql = sql_path.read_text(encoding="utf-8")
    if "select *" in " ".join(sql.lower().split()):
        raise MartBuildError(f"Mart SQL must not use SELECT *: {sql_path}")
    parameters = {f"{table}_path": str(path.resolve()) for table, path in paths.items()}
    with duckdb.connect(database=":memory:") as connection:
        mart = connection.execute(sql, parameters).fetch_df()

    validate_mart(mart, dataset_type, expected_n_users)
    crosscheck_with_pandas(mart, tables)
    return MartBuildResult(
        dataset_type=dataset_type,
        mart=mart,
        input_paths=paths,
        output_path=mart_path(data_root, dataset_type),
        crosscheck_passed=True,
    )


def build_all_marts(config: dict[str, Any], overwrite: bool = False) -> list[MartBuildResult]:
    """Build both marts, writing only after every validation and cross-check passes."""
    data_root = Path(config["data_generation"]["output_dir"])
    outputs = [mart_path(data_root, dataset_type) for dataset_type in DATASET_TYPES]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise MartBuildError(
            f"Processed mart files already exist: {', '.join(existing)}. "
            "Use --overwrite to replace them."
        )

    results = [build_mart_in_memory(config, dataset_type) for dataset_type in DATASET_TYPES]
    for result in results:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.mart.to_parquet(result.output_path, index=False)
    return results


def validate_data_and_write_report(config: dict[str, Any]) -> dict[str, Any]:
    """Validate raw tables and existing marts, then write a machine-readable summary."""
    data_root = Path(config["data_generation"]["output_dir"])
    expected_n_users = int(config["experiment"]["n_users"])
    summaries: dict[str, Any] = {}
    report: dict[str, Any] = {
        "synthetic_data_only": True,
        "checks_passed": False,
        "failed_checks": [],
        "datasets": summaries,
    }
    report_path = Path(config["data_quality"]["report_path"])
    try:
        for dataset_type in DATASET_TYPES:
            paths = raw_paths(data_root, dataset_type)
            tables = _read_raw_tables(paths)
            validate_raw_tables(
                tables["users"],
                tables["assignment"],
                tables["outcomes"],
                dataset_type,
                expected_n_users,
            )
            output = mart_path(data_root, dataset_type)
            if not output.is_file():
                raise DataValidationError(f"Processed mart file is missing: {output}")
            mart = pd.read_parquet(output)
            validate_mart(mart, dataset_type, expected_n_users)
            crosscheck_with_pandas(mart, tables)
            summary = quality_summary(mart, dataset_type)
            summary["sql_pandas_crosscheck"] = True
            summaries[dataset_type] = summary
        report["checks_passed"] = True
    except (DataValidationError, MartBuildError) as error:
        report["failed_checks"] = [str(error)]
        _write_json_report(report_path, report)
        raise

    _write_json_report(report_path, report)
    return report


def _write_json_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
