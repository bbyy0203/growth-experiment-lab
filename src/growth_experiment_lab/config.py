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
        "data_generation",
        "mart",
        "data_quality",
        "experiment",
        "inference",
        "experiment_design",
        "randomization_diagnostics",
        "ab_analysis",
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
    experiment_design = config["experiment_design"]
    diagnostics = config["randomization_diagnostics"]
    ab_analysis = config["ab_analysis"]
    metrics = config["metrics"]
    observational = config["observational"]
    data_generation = config["data_generation"]
    mart = config["mart"]
    data_quality = config["data_quality"]

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
    if inference.get("alternative") != "two-sided":
        raise ConfigError("inference.alternative must be 'two-sided'")

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
    if observational.get("propensity_model") != "logistic_regression":
        raise ConfigError("observational.propensity_model must be 'logistic_regression'")
    if observational.get("main_analysis_truncate_weights") is not False:
        raise ConfigError("observational.main_analysis_truncate_weights must be false")
    for key in ("causal_bootstrap_seed", "causal_bootstrap_resamples"):
        if not isinstance(observational.get(key), int) or observational[key] <= 0:
            raise ConfigError(f"observational.{key} must be a positive integer")
    if observational.get("sensitivity_lower_quantile") != 0.01:
        raise ConfigError("observational sensitivity lower quantile must be 0.01")
    if observational.get("sensitivity_upper_quantile") != 0.99:
        raise ConfigError("observational sensitivity upper quantile must be 0.99")
    for key in (
        "results_path",
        "markdown_path",
        "overlap_plot_path",
        "weight_plot_path",
        "balance_plot_path",
        "simulation_validation_path",
    ):
        if not isinstance(observational.get(key), str):
            raise ConfigError(f"observational.{key} must be a string")

    if not isinstance(data_generation.get("version"), str):
        raise ConfigError("data_generation.version must be a string")
    if not isinstance(data_generation.get("output_dir"), str):
        raise ConfigError("data_generation.output_dir must be a string")
    if not isinstance(mart.get("sql_dir"), str):
        raise ConfigError("mart.sql_dir must be a string")
    if not isinstance(data_quality.get("report_path"), str):
        raise ConfigError("data_quality.report_path must be a string")
    generation = observational.get("generation")
    if not isinstance(generation, dict):
        raise ConfigError("observational.generation must be a table")
    propensity_min = generation.get("propensity_min")
    propensity_max = generation.get("propensity_max")
    if not isinstance(propensity_min, (int, float)) or not 0 < propensity_min < 1:
        raise ConfigError("observational.generation.propensity_min must be between 0 and 1")
    if not isinstance(propensity_max, (int, float)) or not 0 < propensity_max < 1:
        raise ConfigError("observational.generation.propensity_max must be between 0 and 1")
    if propensity_min >= propensity_max:
        raise ConfigError("propensity_min must be smaller than propensity_max")

    if experiment_design.get("primary_metric") != "next_day_retention":
        raise ConfigError("experiment_design.primary_metric must be 'next_day_retention'")
    primary_mde = experiment_design.get("primary_absolute_mde")
    if not isinstance(primary_mde, (int, float)) or primary_mde <= 0:
        raise ConfigError("experiment_design.primary_absolute_mde must be positive")
    if not isinstance(experiment_design.get("report_path"), str):
        raise ConfigError("experiment_design.report_path must be a string")

    srm_alpha = diagnostics.get("srm_alpha")
    if not isinstance(srm_alpha, (int, float)) or not 0 < srm_alpha < 1:
        raise ConfigError("randomization_diagnostics.srm_alpha must be between 0 and 1")
    smd_threshold = diagnostics.get("smd_warning_threshold")
    if not isinstance(smd_threshold, (int, float)) or smd_threshold <= 0:
        raise ConfigError("randomization_diagnostics.smd_warning_threshold must be positive")
    for path_key in ("report_path", "balance_plot_path", "aa_plot_path"):
        if not isinstance(diagnostics.get(path_key), str):
            raise ConfigError(f"randomization_diagnostics.{path_key} must be a string")

    aa_simulation = config["aa_simulation"]
    if not isinstance(aa_simulation.get("random_seed"), int):
        raise ConfigError("aa_simulation.random_seed must be an integer")
    for count_key in ("n_simulations", "n_users_per_simulation"):
        value = aa_simulation.get(count_key)
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"aa_simulation.{count_key} must be a positive integer")
    aa_alpha = aa_simulation.get("alpha")
    if not isinstance(aa_alpha, (int, float)) or not 0 < aa_alpha < 1:
        raise ConfigError("aa_simulation.alpha must be between 0 and 1")

    if not isinstance(ab_analysis.get("bootstrap_seed"), int):
        raise ConfigError("ab_analysis.bootstrap_seed must be an integer")
    bootstrap_resamples = ab_analysis.get("bootstrap_resamples")
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples <= 0:
        raise ConfigError("ab_analysis.bootstrap_resamples must be a positive integer")
    for path_key in (
        "results_path",
        "markdown_path",
        "effect_plot_path",
        "cuped_plot_path",
        "subgroup_plot_path",
        "simulation_validation_path",
    ):
        if not isinstance(ab_analysis.get(path_key), str):
            raise ConfigError(f"ab_analysis.{path_key} must be a string")
