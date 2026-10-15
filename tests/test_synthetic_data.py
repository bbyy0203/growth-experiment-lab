"""Tests for reproducible stage-2A synthetic data generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from growth_experiment_lab.cli import main
from growth_experiment_lab.config import load_config, validate_config
from growth_experiment_lab.synthetic_data import (
    OUTCOME_COLUMNS,
    PROPENSITY_INPUT_COLUMNS,
    USER_COLUMNS,
    DataGenerationError,
    assignment_probabilities,
    generate_all,
    write_generation_result,
)


@pytest.fixture
def small_config() -> dict[str, object]:
    config = load_config()
    config["experiment"]["n_users"] = 2_000
    return config


def test_same_seed_is_exactly_reproducible(small_config: dict[str, object]) -> None:
    first = generate_all(small_config)
    second = generate_all(small_config)

    for dataset_type in ("randomized", "observational"):
        pd.testing.assert_frame_equal(
            first.datasets[dataset_type].users, second.datasets[dataset_type].users
        )
        pd.testing.assert_frame_equal(
            first.datasets[dataset_type].assignment, second.datasets[dataset_type].assignment
        )
        pd.testing.assert_frame_equal(
            first.datasets[dataset_type].outcomes, second.datasets[dataset_type].outcomes
        )
    assert first.metadata == second.metadata


def test_different_seed_changes_data(small_config: dict[str, object]) -> None:
    changed = deepcopy(small_config)
    changed["project"]["random_seed"] += 1  # type: ignore[index]

    first = generate_all(small_config)
    second = generate_all(changed)

    assert not first.datasets["randomized"].assignment.equals(
        second.datasets["randomized"].assignment
    )


@pytest.mark.parametrize("dataset_type", ["randomized", "observational"])
def test_public_table_contracts(
    small_config: dict[str, object], dataset_type: str
) -> None:
    dataset = generate_all(small_config).datasets[dataset_type]  # type: ignore[index]

    assert list(dataset.users.columns) == USER_COLUMNS
    assert list(dataset.outcomes.columns) == OUTCOME_COLUMNS
    assert dataset.users["user_id"].is_unique
    assert dataset.assignment["user_id"].is_unique
    assert len(dataset.users) == len(dataset.assignment) == len(dataset.outcomes) == 2_000
    assert set(dataset.assignment["treatment"].unique()).issubset({0, 1})
    for metric in ("clicked", "next_day_retention", "negative_feedback"):
        assert set(dataset.outcomes[metric].unique()).issubset({0, 1})
    assert dataset.outcomes["watch_time_minutes"].ge(0).all()
    assert 0.10 < dataset.outcomes["watch_time_minutes"].eq(0).mean() < 0.60


def test_randomized_assignment_probability_is_constant(
    small_config: dict[str, object],
) -> None:
    dataset = generate_all(small_config).datasets["randomized"]
    expected_share = small_config["experiment"]["treatment_share"]  # type: ignore[index]

    assert np.allclose(dataset.assignment["assignment_probability"], expected_share)
    changed_users = dataset.users.copy()
    changed_users["pre_experiment_activity_score"] *= -100
    probabilities = assignment_probabilities(changed_users, "randomized", 0.5, 0.05, 0.95)
    assert np.allclose(probabilities, 0.5)


def test_observational_propensity_uses_declared_pretreatment_inputs(
    small_config: dict[str, object],
) -> None:
    dataset = generate_all(small_config).datasets["observational"]
    probabilities = dataset.assignment["assignment_probability"]

    assert probabilities.nunique() > 100
    assert probabilities.between(0.05, 0.95).all()
    assert set(PROPENSITY_INPUT_COLUMNS).issubset(dataset.users.columns)
    assert set(OUTCOME_COLUMNS[1:]).isdisjoint(PROPENSITY_INPUT_COLUMNS)


def test_true_effect_metadata_is_complete_and_recomputed(
    small_config: dict[str, object],
) -> None:
    result = generate_all(small_config)
    expected_fields = {
        "random_seed",
        "n_users",
        "data_generation_version",
        "true_sample_ate",
        "nominal_generation_parameters",
        "probability_clipping",
        "effects_recomputed_after_clipping",
        "actual_treatment_share",
        "config_summary",
    }

    assert expected_fields == set(result.metadata)
    assert result.metadata["effects_recomputed_after_clipping"] is True
    for dataset_type in ("randomized", "observational"):
        effects = result.metadata["true_sample_ate"][dataset_type]
        assert set(effects) == {
            "clicked",
            "watch_time_minutes",
            "next_day_retention",
            "negative_feedback",
        }
        if result.metadata["probability_clipping"][dataset_type]["negative_feedback"]:
            nominal = small_config["metrics"]["negative_feedback"]["treatment_effect"]  # type: ignore[index]
            assert not np.isclose(effects["negative_feedback"], nominal, atol=1e-10)


def test_write_refuses_overwrite_and_allows_explicit_overwrite(
    small_config: dict[str, object], tmp_path: Path
) -> None:
    result = generate_all(small_config)
    paths = write_generation_result(result, tmp_path)
    original_mtime = paths[0].stat().st_mtime_ns

    with pytest.raises(DataGenerationError, match="--overwrite"):
        write_generation_result(result, tmp_path)

    rewritten = write_generation_result(result, tmp_path, overwrite=True)
    assert rewritten == paths
    assert paths[0].stat().st_mtime_ns >= original_mtime


def test_generation_configuration_validation(small_config: dict[str, object]) -> None:
    config = deepcopy(small_config)
    config["observational"]["generation"]["propensity_min"] = 0.99  # type: ignore[index]

    with pytest.raises(ValueError, match="propensity_min"):
        validate_config(config)


def test_cli_requires_overwrite_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_text = Path("configs/default.toml").read_text(encoding="utf-8")
    config_text = config_text.replace("n_users = 100000", "n_users = 200")
    config_text = config_text.replace(
        'output_dir = "data"', f'output_dir = "{tmp_path.as_posix()}"'
    )
    config_path = tmp_path / "test.toml"
    config_path.write_text(config_text, encoding="utf-8")

    assert main(["generate-data", "--config", str(config_path)]) == 0
    with pytest.raises(SystemExit) as error:
        main(["generate-data", "--config", str(config_path)])
    assert error.value.code == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(["generate-data", "--config", str(config_path), "--overwrite"]) == 0
