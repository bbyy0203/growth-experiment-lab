"""Read and validate the project's TOML configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/default.toml")


class ConfigError(ValueError):
    """Raised when a project configuration violates the stage-1 contract."""


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a TOML configuration and validate its top-level statistical contract."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in {config_path}: {error}") from error

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate stable invariants without implementing analytical behavior."""
    required_sections = {
        "project",
        "experiment",
        "inference",
        "metrics",
        "aa_simulation",
        "observational",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(sorted(missing))}")

    for section in required_sections:
        if not isinstance(config[section], dict):
            raise ConfigError(f"Configuration section '{section}' must be a table")

    project = config["project"]
    experiment = config["experiment"]
    inference = config["inference"]
    metrics = config["metrics"]
    observational = config["observational"]

    if not isinstance(project.get("random_seed"), int):
        raise ConfigError("project.random_seed must be an integer")
    if not isinstance(experiment.get("n_users"), int) or experiment["n_users"] <= 0:
        raise ConfigError("experiment.n_users must be a positive integer")
    if not 0 < experiment.get("treatment_share", 0) < 1:
        raise ConfigError("experiment.treatment_share must be between 0 and 1")
    if experiment.get("unit") != "user_id":
        raise ConfigError("experiment.unit must be 'user_id'")
    if experiment.get("estimand") != "intention_to_treat":
        raise ConfigError("experiment.estimand must be 'intention_to_treat'")
    if experiment.get("segment") != "is_new_user":
        raise ConfigError("experiment.segment must be 'is_new_user'")
    if not 0 < inference.get("alpha", 0) < 1:
        raise ConfigError("inference.alpha must be between 0 and 1")
    if not 0 < inference.get("power", 0) < 1:
        raise ConfigError("inference.power must be between 0 and 1")

    expected_roles = {
        "next_day_retention": "primary",
        "clicked": "secondary",
        "watch_time_minutes": "secondary",
        "negative_feedback": "guardrail",
    }
    unexpected_metrics = set(metrics).difference(expected_roles)
    if unexpected_metrics:
        names = ", ".join(sorted(unexpected_metrics))
        raise ConfigError(f"Unsupported metrics: {names}")

    for metric_name, expected_role in expected_roles.items():
        metric = metrics.get(metric_name)
        if not isinstance(metric, dict):
            raise ConfigError(f"Missing metric configuration: {metric_name}")
        if metric.get("role") != expected_role:
            raise ConfigError(f"metrics.{metric_name}.role must be '{expected_role}'")

    roles = [metric["role"] for metric in metrics.values()]
    if roles.count("primary") != 1:
        raise ConfigError("Exactly one metric must have the 'primary' role")

    guardrail_threshold = metrics["negative_feedback"].get("maximum_allowed_increase")
    if not isinstance(guardrail_threshold, (int, float)) or not 0 < guardrail_threshold < 1:
        raise ConfigError("negative_feedback maximum allowed increase must be between 0 and 1")
    if guardrail_threshold != 0.005:
        raise ConfigError("negative_feedback maximum allowed increase must be 0.005")
    if observational.get("estimand") != "ATE":
        raise ConfigError("observational.estimand must be 'ATE'")
    if observational.get("weighting") != "stabilized_iptw":
        raise ConfigError("observational.weighting must be 'stabilized_iptw'")
