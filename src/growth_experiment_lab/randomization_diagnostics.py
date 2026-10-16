"""SRM and pre-treatment covariate balance diagnostics for randomized marts."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "growth_experiment_lab_matplotlib")
)

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from growth_experiment_lab.data_validation import PRETREATMENT_COVARIATES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORICAL_COVARIATES = ["device_type", "region"]
BINARY_COVARIATES = ["is_new_user"]
CONTINUOUS_COVARIATES = [
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
]


class RandomizationDiagnosticError(ValueError):
    """Raised when a randomization diagnostic is invalid or unsafe."""


def srm_check(
    control_count: int,
    treatment_count: int,
    expected_treatment_share: float,
    srm_alpha: float,
    dataset_type: str = "randomized",
) -> dict[str, Any]:
    """Run a chi-square goodness-of-fit sample-ratio-mismatch check."""
    if dataset_type != "randomized":
        raise RandomizationDiagnosticError("SRM is allowed only for randomized data")
    if control_count < 0 or treatment_count < 0 or control_count + treatment_count <= 0:
        raise RandomizationDiagnosticError(
            "group counts must be non-negative with a positive total"
        )
    if not 0 < expected_treatment_share < 1:
        raise RandomizationDiagnosticError("expected_treatment_share must be between 0 and 1")
    if not 0 < srm_alpha < 1:
        raise RandomizationDiagnosticError("srm_alpha must be between 0 and 1")
    total = control_count + treatment_count
    expected_treatment = total * expected_treatment_share
    expected_control = total - expected_treatment
    statistic, p_value = chisquare(
        [control_count, treatment_count], f_exp=[expected_control, expected_treatment]
    )
    passed = bool(p_value >= srm_alpha)
    interpretation = (
        "No sample-ratio mismatch detected; this does not prove randomization is fully correct."
        if passed
        else "Sample-ratio mismatch detected at the configured threshold."
    )
    return {
        "observed_control_count": control_count,
        "observed_treatment_count": treatment_count,
        "observed_treatment_share": treatment_count / total,
        "expected_control_count": expected_control,
        "expected_treatment_count": expected_treatment,
        "expected_treatment_share": expected_treatment_share,
        "chi_square_statistic": float(statistic),
        "p_value": float(p_value),
        "srm_alpha": srm_alpha,
        "passed": passed,
        "interpretation": interpretation,
    }


def continuous_smd(control: pd.Series, treatment: pd.Series) -> dict[str, Any]:
    """Calculate a pooled-SD standardized mean difference for a numeric covariate."""
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    difference = treatment_mean - control_mean
    pooled_sd = math.sqrt((float(control.var(ddof=1)) + float(treatment.var(ddof=1))) / 2.0)
    if pooled_sd == 0:
        smd = 0.0 if difference == 0 else math.copysign(math.inf, difference)
        status = "constant_equal" if difference == 0 else "zero_variance_mean_difference"
    else:
        smd = difference / pooled_sd
        status = "ok"
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "raw_difference": difference,
        "pooled_standard_deviation": pooled_sd,
        "smd": smd,
        "absolute_smd": abs(smd),
        "status": status,
    }


def binary_smd(control: pd.Series, treatment: pd.Series) -> dict[str, Any]:
    """Calculate a pooled-Bernoulli standardized difference for a binary covariate."""
    if not bool(control.isin([0, 1]).all()) or not bool(treatment.isin([0, 1]).all()):
        raise RandomizationDiagnosticError("binary SMD inputs must contain only 0/1")
    control_proportion = float(control.mean())
    treatment_proportion = float(treatment.mean())
    difference = treatment_proportion - control_proportion
    pooled_sd = math.sqrt(
        (
            control_proportion * (1.0 - control_proportion)
            + treatment_proportion * (1.0 - treatment_proportion)
        )
        / 2.0
    )
    if pooled_sd == 0:
        smd = 0.0 if difference == 0 else math.copysign(math.inf, difference)
        status = "constant_equal" if difference == 0 else "zero_variance_mean_difference"
    else:
        smd = difference / pooled_sd
        status = "ok"
    return {
        "control_proportion": control_proportion,
        "treatment_proportion": treatment_proportion,
        "proportion_difference": difference,
        "pooled_bernoulli_standard_deviation": pooled_sd,
        "smd": smd,
        "absolute_smd": abs(smd),
        "status": status,
    }


def balance_diagnostics(
    mart: pd.DataFrame,
    warning_threshold: float,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Diagnose only allowlisted pre-treatment covariates in a randomized mart."""
    if set(mart["dataset_type"].astype(str).unique()) != {"randomized"}:
        raise RandomizationDiagnosticError("balance diagnostics require only randomized data")
    if warning_threshold <= 0:
        raise RandomizationDiagnosticError("warning_threshold must be positive")
    requested = PRETREATMENT_COVARIATES if covariates is None else covariates
    rejected = set(requested).difference(PRETREATMENT_COVARIATES)
    if rejected:
        raise RandomizationDiagnosticError(
            f"balance covariates are not in the pre-treatment whitelist: {sorted(rejected)}"
        )
    missing = set(requested).difference(mart.columns)
    if missing:
        raise RandomizationDiagnosticError(f"mart is missing balance covariates: {sorted(missing)}")
    control = mart.loc[mart["treatment"].eq(0)]
    treatment = mart.loc[mart["treatment"].eq(1)]
    if control.empty or treatment.empty:
        raise RandomizationDiagnosticError("both randomized groups are required for balance checks")

    entries: list[dict[str, Any]] = []
    variable_maximums: dict[str, float] = {}
    for variable in requested:
        if variable in CATEGORICAL_COVARIATES:
            levels = sorted(
                set(control[variable].astype(str)).union(treatment[variable].astype(str))
            )
            level_values = []
            for level in levels:
                values = binary_smd(
                    control[variable].astype(str).eq(level).astype(int),
                    treatment[variable].astype(str).eq(level).astype(int),
                )
                entry = {"variable": variable, "level": level, "label": f"{variable}={level}"}
                entry.update(values)
                entries.append(entry)
                level_values.append(values["absolute_smd"])
            variable_maximums[variable] = max(level_values)
        elif variable in BINARY_COVARIATES:
            values = binary_smd(control[variable], treatment[variable])
            entry = {"variable": variable, "level": None, "label": variable}
            entry.update(values)
            entries.append(entry)
            variable_maximums[variable] = values["absolute_smd"]
        else:
            values = continuous_smd(control[variable], treatment[variable])
            entry = {"variable": variable, "level": None, "label": variable}
            entry.update(values)
            entries.append(entry)
            variable_maximums[variable] = values["absolute_smd"]

    warning_labels = [
        entry["label"] for entry in entries if entry["absolute_smd"] > warning_threshold
    ]
    return {
        "covariate_whitelist": list(requested),
        "entries": entries,
        "variable_maximum_absolute_smd": variable_maximums,
        "maximum_absolute_smd": max(entry["absolute_smd"] for entry in entries),
        "warning_threshold": warning_threshold,
        "warning_variables": warning_labels,
        "has_warning": bool(warning_labels),
        "interpretation": (
            "Threshold exceedances are warnings, not proof that the experiment is invalid."
        ),
    }


def save_balance_plot(balance: dict[str, Any], output_path: str | Path) -> Path:
    """Save an absolute-SMD plot containing only pre-treatment covariates."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = balance["entries"]
    labels = [entry["label"] for entry in entries]
    values = [entry["absolute_smd"] for entry in entries]
    figure_height = max(4.0, 0.38 * len(entries))
    figure, axis = plt.subplots(figsize=(9, figure_height))
    positions = np.arange(len(entries))
    axis.barh(positions, values, color="#4C78A8")
    axis.axvline(balance["warning_threshold"], color="#E45756", linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Absolute standardized mean difference")
    axis.set_title("Synthetic randomized data: pre-treatment balance")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
