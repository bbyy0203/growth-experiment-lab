"""Formal randomized-experiment analysis without access to simulation truth."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "growth_experiment_lab_matplotlib")
)

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from growth_experiment_lab.ab_effects import (  # noqa: E402
    analyze_binary_metric,
    analyze_continuous_metric,
    apply_secondary_holm,
    guardrail_status,
)
from growth_experiment_lab.bootstrap import bootstrap_mean_difference  # noqa: E402
from growth_experiment_lab.cuped import analyze_watch_time_cuped  # noqa: E402
from growth_experiment_lab.data_validation import validate_mart  # noqa: E402
from growth_experiment_lab.mart import mart_path  # noqa: E402
from growth_experiment_lab.randomization_diagnostics import (  # noqa: E402
    balance_diagnostics,
    srm_check,
)
from growth_experiment_lab.subgroup_analysis import (  # noqa: E402
    analyze_predefined_subgroups,
)

METRIC_TYPES = {
    "next_day_retention": "binary",
    "clicked": "binary",
    "watch_time_minutes": "continuous",
    "negative_feedback": "binary",
}


class ABAnalysisError(RuntimeError):
    """Raised when formal randomized A/B analysis cannot proceed safely."""


def _output_paths(config: dict[str, Any], skip_cuped: bool, skip_segments: bool) -> list[Path]:
    settings = config["ab_analysis"]
    paths = [
        Path(settings["results_path"]),
        Path(settings["markdown_path"]),
        Path(settings["effect_plot_path"]),
    ]
    if not skip_cuped:
        paths.append(Path(settings["cuped_plot_path"]))
    if not skip_segments:
        paths.append(Path(settings["subgroup_plot_path"]))
    return paths


def _refuse_existing(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise ABAnalysisError(
            f"Output files already exist: {', '.join(existing)}. Use --overwrite to replace them."
        )


def _read_randomized_mart(config: dict[str, Any]) -> pd.DataFrame:
    path = mart_path(Path(config["data_generation"]["output_dir"]), "randomized")
    if not path.is_file():
        raise ABAnalysisError(f"Randomized mart does not exist: {path}")
    mart = pd.read_parquet(path)
    validate_mart(mart, "randomized", int(config["experiment"]["n_users"]))
    return mart


def analyze_ab(
    config: dict[str, Any],
    overwrite: bool = False,
    skip_bootstrap: bool = False,
    skip_cuped: bool = False,
    skip_segments: bool = False,
) -> dict[str, Any]:
    """Run the predeclared ITT analysis and write JSON, Markdown, and plots."""
    paths = _output_paths(config, skip_cuped, skip_segments)
    _refuse_existing(paths, overwrite)
    mart = _read_randomized_mart(config)
    diagnostics = config["randomization_diagnostics"]
    treatment_n = int(mart["treatment"].sum())
    control_n = len(mart) - treatment_n
    srm = srm_check(
        control_n,
        treatment_n,
        float(config["experiment"]["treatment_share"]),
        float(diagnostics["srm_alpha"]),
    )
    if not srm["passed"]:
        raise ABAnalysisError("SRM failed; formal A/B analysis was not produced")
    balance = balance_diagnostics(mart, float(diagnostics["smd_warning_threshold"]))

    confidence_level = float(config["inference"]["confidence_level"])
    alpha = float(config["inference"]["alpha"])
    metric_results: dict[str, dict[str, Any]] = {}
    for metric_name, metric_type in METRIC_TYPES.items():
        if metric_type == "binary":
            result = analyze_binary_metric(mart, metric_name, confidence_level)
        else:
            result = analyze_continuous_metric(mart, metric_name, confidence_level)
        metric_results[metric_name] = result
    apply_secondary_holm(metric_results, alpha)

    if skip_bootstrap:
        for result in metric_results.values():
            result["bootstrap"] = {"skipped": True}
    else:
        base_seed = int(config["ab_analysis"]["bootstrap_seed"])
        resamples = int(config["ab_analysis"]["bootstrap_resamples"])
        for index, (metric_name, metric_type) in enumerate(METRIC_TYPES.items()):
            metric_results[metric_name]["bootstrap"] = bootstrap_mean_difference(
                mart,
                metric_name,
                metric_type,  # type: ignore[arg-type]
                resamples,
                base_seed + index,
                confidence_level,
            )

    guardrail_threshold = float(
        config["metrics"]["negative_feedback"]["maximum_allowed_increase"]
    )
    guardrail = metric_results["negative_feedback"]
    guardrail["maximum_allowed_increase"] = guardrail_threshold
    guardrail["guardrail_status"] = guardrail_status(guardrail, guardrail_threshold)
    guardrail["business_interpretation"] = (
        f"Guardrail status is {guardrail['guardrail_status']} against the "
        f"+{guardrail_threshold:.3f} "
        "absolute threshold; positive effects are harmful."
    )

    cuped: dict[str, Any]
    if skip_cuped:
        cuped = {"skipped": True}
    else:
        cuped = analyze_watch_time_cuped(mart, confidence_level)

    segments: dict[str, Any]
    if skip_segments:
        segments = {"skipped": True}
    else:
        segments = {
            metric_name: analyze_predefined_subgroups(
                mart,
                metric_name,
                metric_type,  # type: ignore[arg-type]
                alpha,
                confidence_level,
            )
            for metric_name, metric_type in METRIC_TYPES.items()
        }

    decision_support = _decision_support(
        metric_results, guardrail["guardrail_status"], cuped, segments, balance, alpha
    )
    settings = config["ab_analysis"]
    generated_plots = {"overall_effects": settings["effect_plot_path"]}
    _save_effect_plot(metric_results, Path(settings["effect_plot_path"]))
    if not skip_cuped:
        _save_cuped_plot(cuped, Path(settings["cuped_plot_path"]))
        generated_plots["cuped"] = settings["cuped_plot_path"]
    if not skip_segments:
        _save_subgroup_plot(segments, Path(settings["subgroup_plot_path"]))
        generated_plots["subgroups"] = settings["subgroup_plot_path"]

    report = {
        "synthetic_data_only": True,
        "not_a_production_experiment": True,
        "analysis_principle": "user-level intention-to-treat using first randomized assignment",
        "dataset_type": "randomized",
        "randomization_quality": {
            "data_validation_passed": True,
            "srm": srm,
            "balance_maximum_absolute_smd": balance["maximum_absolute_smd"],
            "balance_warning_variables": balance["warning_variables"],
        },
        "metric_roles": {
            "primary": ["next_day_retention"],
            "secondary_holm_family": ["clicked", "watch_time_minutes"],
            "guardrail": ["negative_feedback"],
        },
        "metrics": metric_results,
        "cuped": cuped,
        "predefined_subgroups": segments,
        "decision_support": decision_support,
        "generated_plots": generated_plots,
        "requires_human_review": True,
    }
    results_path = Path(settings["results_path"])
    markdown_path = Path(settings["markdown_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _primary_status(result: dict[str, Any], alpha: float) -> str:
    effect = result["absolute_effect"]
    if effect < 0:
        return "negative"
    if effect == 0:
        return "no_improvement"
    if result["raw_p_value"] < alpha:
        return "positive_significant"
    return "positive_inconclusive"


def _decision_support(
    metrics: dict[str, dict[str, Any]],
    guardrail: str,
    cuped: dict[str, Any],
    segments: dict[str, Any],
    balance: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    secondary = {}
    for metric in ("clicked", "watch_time_minutes"):
        result = metrics[metric]
        secondary[metric] = {
            "direction": "positive" if result["absolute_effect"] > 0 else "negative",
            "holm_significant": bool(result["adjusted_p_value"] < alpha),
        }
    open_questions = ["Human review is required before any product decision."]
    if guardrail == "inconclusive":
        open_questions.append("The guardrail interval crosses the allowed-harm threshold.")
    if balance["has_warning"]:
        open_questions.append("Review pre-treatment balance warnings before interpretation.")
    return {
        "randomization_quality": {
            "data_validation_passed": True,
            "srm_passed": True,
            "balance_warnings": balance["warning_variables"],
        },
        "primary_metric_status": _primary_status(metrics["next_day_retention"], alpha),
        "secondary_metrics_status": secondary,
        "guardrail_status": guardrail,
        "cuped_summary": cuped,
        "subgroup_summary": segments,
        "open_questions": open_questions,
        "requires_human_review": True,
        "decision_boundary": "This project does not automatically recommend launch or stop.",
    }


def _save_effect_plot(metrics: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(METRIC_TYPES)
    effects = [metrics[name]["absolute_effect"] for name in names]
    lower = [metrics[name]["confidence_interval_lower"] for name in names]
    upper = [metrics[name]["confidence_interval_upper"] for name in names]
    errors = [[effect - low for effect, low in zip(effects, lower, strict=True)],
              [high - effect for effect, high in zip(effects, upper, strict=True)]]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(effects, range(len(names)), xerr=errors, fmt="o", capsize=4)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("ITT absolute effect (metric-native units)")
    axis.set_title("Synthetic randomized experiment: overall effects")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_cuped_plot(cuped: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    effects = [cuped["raw_absolute_effect"], cuped["cuped_absolute_effect"]]
    errors = [cuped["raw_standard_error"], cuped["cuped_standard_error"]]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(effects, [0, 1], xerr=[1.96 * value for value in errors], fmt="o", capsize=4)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks([0, 1], ["Raw", "CUPED"])
    axis.set_xlabel("Watch-time mean difference")
    axis.set_title("Synthetic data: raw and CUPED precision")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_subgroup_plot(segments: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = segments["next_day_retention"]["subgroup_effects"]
    labels = ["existing_users", "new_users"]
    effects = [primary[label]["absolute_effect"] for label in labels]
    lower = [primary[label]["confidence_interval_lower"] for label in labels]
    upper = [primary[label]["confidence_interval_upper"] for label in labels]
    errors = [[effect - low for effect, low in zip(effects, lower, strict=True)],
              [high - effect for effect, high in zip(effects, upper, strict=True)]]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.errorbar(effects, [0, 1], xerr=errors, fmt="o", capsize=4)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Next-day-retention risk difference")
    axis.set_title("Predefined synthetic-data subgroup effects")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Synthetic randomized A/B analysis",
        "",
        "> All data are synthetic. This is not a company production experiment or "
        "real product effect.",
        "> The report does not automatically decide launch; human review is required.",
        "",
        "## Randomization quality",
        "",
        f"- Data validation passed: {report['randomization_quality']['data_validation_passed']}",
        f"- SRM passed: {report['randomization_quality']['srm']['passed']}",
        f"- Balance warnings: {report['randomization_quality']['balance_warning_variables']}",
        "",
        "## Overall ITT effects",
        "",
        "| Metric | Control | Treatment | Absolute effect | 95% CI | Raw p | Adjusted p | "
        "Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, result in report["metrics"].items():
        control = result.get("control_rate", result.get("control_mean"))
        treatment = result.get("treatment_rate", result.get("treatment_mean"))
        adjusted = result["adjusted_p_value"]
        adjusted_text = "N/A" if adjusted is None else f"{adjusted:.6g}"
        bootstrap = result["bootstrap"]
        bootstrap_text = (
            "Skipped"
            if bootstrap.get("skipped")
            else (
                f"[{bootstrap['percentile_confidence_interval_lower']:.6f}, "
                f"{bootstrap['percentile_confidence_interval_upper']:.6f}]"
            )
        )
        lines.append(
            f"| {metric} | {control:.6f} | {treatment:.6f} | "
            f"{result['absolute_effect']:.6f} | "
            f"[{result['confidence_interval_lower']:.6f}, "
            f"{result['confidence_interval_upper']:.6f}] | "
            f"{result['raw_p_value']:.6g} | {adjusted_text} | {bootstrap_text} |"
        )
    cuped = report["cuped"]
    if cuped.get("skipped"):
        cuped_lines = ["CUPED was skipped by request."]
    else:
        cuped_lines = [
            f"- Theta: {cuped['theta']:.6f}",
            f"- Raw effect: {cuped['raw_absolute_effect']:.6f}",
            f"- CUPED effect: {cuped['cuped_absolute_effect']:.6f}",
            f"- Raw/CUPED standard error: {cuped['raw_standard_error']:.6f} / "
            f"{cuped['cuped_standard_error']:.6f}",
            f"- Variance reduction: {cuped['variance_reduction_percent']:.3f}%",
            f"- CUPED 95% CI: [{cuped['confidence_interval_lower']:.6f}, "
            f"{cuped['confidence_interval_upper']:.6f}]",
            f"- CUPED p-value: {cuped['p_value']:.6g}",
        ]
    segments = report["predefined_subgroups"]
    if segments.get("skipped"):
        segment_lines = ["Predefined subgroup analysis was skipped by request."]
    else:
        segment_lines = [
            "| Metric | Existing-user effect | New-user effect | Interaction coefficient | "
            "Interaction p | Heterogeneity evidence |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for metric, result in segments.items():
            existing = result["subgroup_effects"]["existing_users"]
            new = result["subgroup_effects"]["new_users"]
            interaction = result["interaction"]
            segment_lines.append(
                f"| {metric} | {existing['absolute_effect']:.6f} "
                f"(n={existing['total_n']}) | {new['absolute_effect']:.6f} "
                f"(n={new['total_n']}) | {interaction['coefficient']:.6f} | "
                f"{interaction['p_value']:.6g} | "
                f"{interaction['evidence_of_heterogeneity']} |"
            )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            f"Status: **{report['metrics']['negative_feedback']['guardrail_status']}** against "
            "the predeclared +0.005 absolute threshold.",
            "",
            "## CUPED",
            "",
            "CUPED uses only pre-experiment watch time and one theta estimated on all users.",
            *cuped_lines,
            "",
            "## Predefined subgroup analysis",
            "",
            "Only `is_new_user` is analyzed. Heterogeneity uses HC3 interaction tests, not a "
            "comparison of separate subgroup p-values.",
            "",
            *segment_lines,
            "",
            "## Decision support and limitations",
            "",
            f"- Primary status: {report['decision_support']['primary_metric_status']}",
            f"- Guardrail status: {report['decision_support']['guardrail_status']}",
            "- Bootstrap is a sensitivity analysis and does not replace the primary tests.",
            "- Subgroup analyses have additional multiplicity and do not trigger "
            "automatic decisions.",
            "- Results cannot be generalized to real users or products.",
            f"- Open questions: {report['decision_support']['open_questions']}",
            "- `requires_human_review = true`",
            "",
        ]
    )
    return "\n".join(lines)
