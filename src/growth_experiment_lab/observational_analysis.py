"""Formal observational IPTW analysis; synthetic truth is deliberately inaccessible here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from growth_experiment_lab.causal_bootstrap import bootstrap_iptw
from growth_experiment_lab.data_validation import validate_mart
from growth_experiment_lab.iptw import balance_table, estimate_all_effects, summarize_balance
from growth_experiment_lab.mart import mart_path
from growth_experiment_lab.propensity import (
    OUTCOMES,
    distribution_summary,
    fit_propensity,
    propensity_diagnostics,
    stabilized_weights,
    trim_weights,
    weight_diagnostics,
)


class ObservationalAnalysisError(RuntimeError):
    """Raised when formal observational analysis cannot run safely."""


def _ensure_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ObservationalAnalysisError(
            f"Outputs already exist: {', '.join(existing)}. Use --overwrite."
        )


def _plots(
    frame: pd.DataFrame,
    scores: np.ndarray,
    weights: np.ndarray,
    balances: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    t = frame["treatment"].to_numpy() == 1
    for key in ("overlap_plot_path", "weight_plot_path", "balance_plot_path"):
        Path(settings[key]).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[~t], bins=40, alpha=0.55, density=True, label="control")
    ax.hist(scores[t], bins=40, alpha=0.55, density=True, label="treatment")
    ax.set(
        xlabel="Estimated propensity score",
        ylabel="Density",
        title="Propensity overlap (synthetic data)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(settings["overlap_plot_path"], dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(weights, bins=50)
    ax.set(xlabel="Stabilized IPTW", ylabel="Users", title="IPTW distribution (synthetic data)")
    fig.tight_layout()
    fig.savefig(settings["weight_plot_path"], dpi=140)
    plt.close(fig)
    rows = balances["before"]["rows"]
    labels = [row["covariate"] for row in rows]
    lookup = {
        stage: {r["covariate"]: r["absolute_smd"] for r in value["rows"]}
        for stage, value in balances.items()
    }
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(5, len(labels) * 0.35)))
    ax.scatter([lookup["before"][x] for x in labels], y, label="before")
    ax.scatter([lookup["iptw"][x] for x in labels], y, label="IPTW")
    ax.scatter([lookup["trimmed_iptw"][x] for x in labels], y, label="trimmed")
    ax.axvline(0.1, color="red", linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Absolute SMD", title="Covariate balance Love plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(settings["balance_plot_path"], dpi=140)
    plt.close(fig)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observational IPTW analysis",
        "",
        (
            "> Synthetic data only. This is a portfolio study, not evidence from a real "
            "product or production system."
        ),
        "",
        "## Causal question",
        "",
        (
            "The estimand is the ATE. Treatment-minus-control effects are adjusted only "
            "for measured pre-treatment covariates."
        ),
        "",
        "## Propensity, overlap, and weights",
        "",
        f"- Model: {report['propensity_model']['method']}",
        f"- Features: {', '.join(report['propensity_model']['features'])}",
        (
            f"- Propensity range: {report['overlap']['overall']['min']:.4f} to "
            f"{report['overlap']['overall']['max']:.4f}"
        ),
        f"- Overall ESS: {report['weights']['effective_sample_size']['overall']:.1f}",
        f"- Warnings: {report['warnings'] or ['none']}",
        "",
        "## Balance",
        "",
        (
            f"Maximum |SMD| before / IPTW / trimmed: "
            f"{report['balance']['before']['maximum_absolute_smd']:.4f} / "
            f"{report['balance']['iptw']['maximum_absolute_smd']:.4f} / "
            f"{report['balance']['trimmed_iptw']['maximum_absolute_smd']:.4f}."
        ),
        "",
        "## Effect estimates",
        "",
        "| Metric | Naive | IPTW ATE | 95% bootstrap CI | Trimmed IPTW |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, value in report["effects"].items():
        bootstrap = value.get("bootstrap")
        interval = (
            "skipped"
            if bootstrap is None
            else (
                f"[{bootstrap['confidence_interval_lower']:.4f}, "
                f"{bootstrap['confidence_interval_upper']:.4f}]"
            )
        )
        lines.append(
            f"| {metric} | {value['naive_effect']:.4f} | {value['iptw_ate']:.4f} | "
            f"{interval} | {value['trimmed_iptw_ate']:.4f} |"
        )
    lines += [
        "",
        "## Assumptions and limitations",
        "",
        (
            "Causal interpretation requires consistency, no interference, positivity, "
            "correct propensity specification, and no unmeasured confounding. IPTW cannot "
            "remove unmeasured confounding. Weight trimming is a predeclared sensitivity "
            "analysis and may change the effective target population."
        ),
        "",
        "Human review is required; this report does not automate a launch decision.",
        "",
    ]
    return "\n".join(lines)


def analyze_observational(
    config: dict[str, Any], overwrite: bool = False, skip_bootstrap: bool = False
) -> dict[str, Any]:
    settings = config["observational"]
    outputs = [
        Path(settings[x])
        for x in (
            "results_path",
            "markdown_path",
            "overlap_plot_path",
            "weight_plot_path",
            "balance_plot_path",
        )
    ]
    _ensure_outputs(outputs, overwrite)
    path = mart_path(Path(config["data_generation"]["output_dir"]), "observational")
    if not path.is_file():
        raise ObservationalAnalysisError(f"Observational mart not found: {path}")
    frame = pd.read_parquet(path)
    validate_mart(frame, "observational", int(config["experiment"]["n_users"]))
    scores, model = fit_propensity(frame)
    overlap = propensity_diagnostics(frame, scores, float(settings["positivity_threshold"]))
    if overlap["severe_violation"]:
        raise ObservationalAnalysisError(
            "Severe positivity violation; causal estimates are not reliable"
        )
    weights = stabilized_weights(frame["treatment"], scores)
    trimmed, low, high = trim_weights(
        weights, settings["sensitivity_lower_quantile"], settings["sensitivity_upper_quantile"]
    )
    balances = {
        "before": summarize_balance(balance_table(frame)),
        "iptw": summarize_balance(balance_table(frame, weights)),
        "trimmed_iptw": summarize_balance(balance_table(frame, trimmed)),
    }
    effects = estimate_all_effects(frame, weights, trimmed)
    bootstrap = (
        None
        if skip_bootstrap
        else bootstrap_iptw(
            frame, settings["causal_bootstrap_resamples"], settings["causal_bootstrap_seed"]
        )
    )
    for metric in OUTCOMES:
        effects[metric]["bootstrap"] = None if bootstrap is None else bootstrap[metric]
    warnings = list(overlap["warnings"])
    if balances["iptw"]["maximum_absolute_smd"] >= balances["before"]["maximum_absolute_smd"]:
        warnings.append("IPTW did not improve maximum absolute SMD")
    report = {
        "synthetic_data_only": True,
        "not_a_production_analysis": True,
        "dataset_type": "observational",
        "estimand": "ATE",
        "treatment": "treatment",
        "propensity_model": model,
        "overlap": overlap,
        "weights": weight_diagnostics(frame, weights),
        "trimmed_weights": {
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "lower_cutoff": low,
            "upper_cutoff": high,
            "summary": distribution_summary(trimmed),
            "diagnostics": weight_diagnostics(frame, trimmed),
        },
        "balance": balances,
        "effects": effects,
        "warnings": warnings,
        "bootstrap_skipped": skip_bootstrap,
        "requires_human_review": True,
        "assumptions": [
            "consistency",
            "no interference",
            "positivity",
            "correct propensity model",
            "no unmeasured confounding",
        ],
        "generated_plots": [str(x) for x in outputs[2:]],
    }
    _plots(frame, scores, weights, balances, settings)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[1].write_text(_markdown(report), encoding="utf-8")
    return report
