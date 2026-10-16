"""Orchestration for prospective design and randomized-data diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from growth_experiment_lab.aa_simulation import run_aa_simulation, save_pvalue_histogram
from growth_experiment_lab.data_validation import validate_mart
from growth_experiment_lab.experiment_design import binary_mde, binary_sample_size
from growth_experiment_lab.mart import mart_path
from growth_experiment_lab.randomization_diagnostics import (
    balance_diagnostics,
    save_balance_plot,
    srm_check,
)


class Stage3OutputError(RuntimeError):
    """Raised when a stage-3 input or output cannot be used safely."""


def _refuse_existing(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise Stage3OutputError(
            f"Output files already exist: {', '.join(existing)}. Use --overwrite to replace them."
        )


def _read_randomized_mart(config: dict[str, Any]) -> pd.DataFrame:
    data_root = Path(config["data_generation"]["output_dir"])
    path = mart_path(data_root, "randomized")
    if not path.is_file():
        raise Stage3OutputError(f"Randomized mart does not exist: {path}")
    mart = pd.read_parquet(path)
    validate_mart(mart, "randomized", int(config["experiment"]["n_users"]))
    return mart


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plan_experiment(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    """Plan the binary primary metric and calculate the current mart's MDE."""
    output_path = Path(config["experiment_design"]["report_path"])
    _refuse_existing([output_path], overwrite)
    mart = _read_randomized_mart(config)
    primary_metric = config["experiment_design"]["primary_metric"]
    metric = config["metrics"][primary_metric]
    baseline = float(metric["control_baseline"])
    effect = float(config["experiment_design"]["primary_absolute_mde"])
    alpha = float(config["inference"]["alpha"])
    power = float(config["inference"]["power"])
    treatment_share = float(config["experiment"]["treatment_share"])
    alternative = config["inference"]["alternative"]
    planned = binary_sample_size(
        baseline, effect, alpha, power, treatment_share, alternative
    )
    treatment_n = int(mart["treatment"].sum())
    control_n = len(mart) - treatment_n
    current_mde = binary_mde(
        baseline, control_n, treatment_n, alpha, power, alternative
    )
    report = {
        "synthetic_data_only": True,
        "primary_metric": primary_metric,
        "planned_design": planned,
        "current_randomized_mart": {
            "control_n": control_n,
            "treatment_n": treatment_n,
            "total_n": len(mart),
            "detectable_absolute_effect": current_mde["detectable_absolute_effect"],
            "detectable_relative_lift": current_mde["detectable_relative_lift"],
            "implied_treatment_probability": current_mde["implied_treatment_probability"],
        },
        "limitations": [
            "Prospective normal-approximation planning is not a treatment-effect estimate.",
            "The baseline and MDE must be chosen before examining experiment outcomes.",
        ],
    }
    _write_json(output_path, report)
    return report


def diagnose_randomization(
    config: dict[str, Any], overwrite: bool = False, skip_aa: bool = False
) -> dict[str, Any]:
    """Run SRM, pre-treatment balance, and optional independent null A/A simulation."""
    settings = config["randomization_diagnostics"]
    report_path = Path(settings["report_path"])
    balance_path = Path(settings["balance_plot_path"])
    aa_path = Path(settings["aa_plot_path"])
    output_paths = [report_path, balance_path] + ([] if skip_aa else [aa_path])
    _refuse_existing(output_paths, overwrite)
    mart = _read_randomized_mart(config)
    treatment_n = int(mart["treatment"].sum())
    control_n = len(mart) - treatment_n
    srm = srm_check(
        control_n,
        treatment_n,
        float(config["experiment"]["treatment_share"]),
        float(settings["srm_alpha"]),
    )
    balance = balance_diagnostics(mart, float(settings["smd_warning_threshold"]))
    save_balance_plot(balance, balance_path)

    warnings: list[str] = []
    critical_failures: list[str] = []
    passed_checks: list[str] = []
    if srm["passed"]:
        passed_checks.append("srm")
    else:
        critical_failures.append("srm_failed")
    if balance["has_warning"]:
        warnings.append("pre_treatment_smd_threshold_exceeded")
    else:
        passed_checks.append("pre_treatment_balance_no_threshold_warning")

    aa_summary: dict[str, Any]
    generated_plots = {"balance": str(balance_path)}
    if skip_aa:
        aa_summary = {"skipped": True}
        passed_checks.append("aa_skipped_by_request")
    else:
        aa_settings = config["aa_simulation"]
        primary_metric = config["experiment_design"]["primary_metric"]
        aa_result = run_aa_simulation(
            simulations=int(aa_settings["n_simulations"]),
            users_per_simulation=int(aa_settings["n_users_per_simulation"]),
            alpha=float(aa_settings["alpha"]),
            random_seed=int(aa_settings["random_seed"]),
            baseline_probability=float(config["metrics"][primary_metric]["control_baseline"]),
            treatment_share=float(config["experiment"]["treatment_share"]),
        )
        save_pvalue_histogram(aa_result, aa_path)
        aa_summary = {key: value for key, value in aa_result.items() if key != "p_values"}
        generated_plots["aa_pvalues"] = str(aa_path)
        if aa_summary["within_expected_interval"]:
            passed_checks.append("aa_false_positive_rate_within_95_percent_interval")
        else:
            warnings.append("aa_false_positive_rate_outside_95_percent_interval")

    report = {
        "synthetic_data_only": True,
        "dataset_type": "randomized",
        "critical_failures": critical_failures,
        "warnings": warnings,
        "passed_checks": passed_checks,
        "srm": srm,
        "pre_treatment_balance": balance,
        "aa_simulation": aa_summary,
        "generated_plots": generated_plots,
        "scope_note": "Diagnostics only; no treatment-effect estimation was performed.",
    }
    _write_json(report_path, report)
    return report
