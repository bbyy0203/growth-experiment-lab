"""Tests for the stage-1 statistical configuration contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from growth_experiment_lab.config import ConfigError, load_config, validate_config


@pytest.fixture
def valid_config() -> dict[str, object]:
    return load_config()


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("experiment", "treatment_share", 0.0, "treatment_share"),
        ("experiment", "treatment_share", 1.1, "treatment_share"),
        ("experiment", "n_users", 0, "n_users"),
        ("inference", "alpha", 1.0, "alpha"),
        ("inference", "power", -0.1, "power"),
        ("observational", "estimand", "ATT", "estimand"),
    ],
)
def test_invalid_scalar_contract_is_rejected(
    valid_config: dict[str, object],
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    config = deepcopy(valid_config)
    config[section][key] = value  # type: ignore[index]

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_missing_metric_is_rejected(valid_config: dict[str, object]) -> None:
    config = deepcopy(valid_config)
    del config["metrics"]["clicked"]  # type: ignore[index]

    with pytest.raises(ConfigError, match="clicked"):
        validate_config(config)


def test_duplicate_metric_role_is_rejected(valid_config: dict[str, object]) -> None:
    config = deepcopy(valid_config)
    config["metrics"]["clicked"]["role"] = "primary"  # type: ignore[index]

    with pytest.raises(ConfigError, match="clicked"):
        validate_config(config)


def test_extra_metric_is_rejected(valid_config: dict[str, object]) -> None:
    config = deepcopy(valid_config)
    config["metrics"]["invented_metric"] = {"role": "primary"}  # type: ignore[index]

    with pytest.raises(ConfigError, match="invented_metric"):
        validate_config(config)


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0, "invalid"])
def test_invalid_guardrail_threshold_is_rejected(
    valid_config: dict[str, object], threshold: object
) -> None:
    config = deepcopy(valid_config)
    config["metrics"]["negative_feedback"]["maximum_allowed_increase"] = threshold  # type: ignore[index]

    with pytest.raises(ConfigError, match="maximum allowed increase"):
        validate_config(config)
