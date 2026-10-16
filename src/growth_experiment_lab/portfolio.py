"""Build the public portfolio report and export lightweight synthetic examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


class PortfolioError(RuntimeError):
    """Raised when portfolio artifacts cannot be built safely."""


def _read_json(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PortfolioError(f"Required report does not exist: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def build_portfolio_report(config: dict[str, Any], overwrite: bool = False) -> Path:
    """Combine formal reports without reading isolated simulation truth."""
    output = Path(config["portfolio"]["report_path"])
    if output.exists() and not overwrite:
        raise PortfolioError(f"Output already exists: {output}. Use --overwrite.")
    quality = _read_json(config["data_quality"]["report_path"])
    design = _read_json(config["experiment_design"]["report_path"])
    diagnostics = _read_json(config["randomization_diagnostics"]["report_path"])
    randomized = _read_json(config["ab_analysis"]["results_path"])
    observational = _read_json(config["observational"]["results_path"])
    aa = diagnostics["aa_simulation"]
    aa_text = (
        "skipped"
        if aa.get("skipped")
        else (
            f"{aa['empirical_false_positive_rate']:.4f} "
            f"({aa['false_positive_count']}/{aa['simulations']})"
        )
    )
    planned = design["planned_design"]
    current = design["current_randomized_mart"]
    lines = [
        "# Growth Experiment Lab — Portfolio Report",
        "",
        "> Generated from synthetic data. This is a personal open-source portfolio project,",
        "> not a company project or production experiment platform.",
        f"> Generated with `configs/{config['portfolio']['config_label']}.toml`.",
        "",
        "`requires_human_review = true`",
        "",
        "## Project scope",
        "",
        "Reproducible user-growth experiment design, randomized A/B evaluation, and",
        "observational stabilized-IPTW evaluation using Python, SQL, and DuckDB.",
        "",
        "## Analysis flow",
        "",
        "```mermaid",
        "flowchart LR",
        "  C[Configuration] --> D[Synthetic data] --> M[DuckDB user mart]",
        "  M --> Q[Data quality] --> E[Design and randomization checks]",
        "  E --> A[Randomized A/B analysis]",
        "  Q --> O[Observational IPTW analysis]",
        "  A --> R[Portfolio report]",
        "  O --> R",
        "```",
        "",
        "## Data contract and quality",
        "",
        f"- Quality checks passed: {quality['checks_passed']}",
        f"- Randomized users: {quality['datasets']['randomized']['total_users']}",
        f"- Observational users: {quality['datasets']['observational']['total_users']}",
        "- Experimental unit: `user_id`; randomized analysis uses first-assignment ITT.",
        "",
        "## Experiment design",
        "",
        f"- Primary metric: `{design['primary_metric']}`",
        f"- Planned sample size: {planned['total_n']}",
        f"- Planned absolute MDE: {planned['absolute_effect']:.4f}",
        f"- Current-mart detectable absolute effect: {current['detectable_absolute_effect']:.4f}",
        "",
        "## Randomization quality",
        "",
        f"- SRM passed: {diagnostics['srm']['passed']}",
        (
            "- Maximum pre-treatment |SMD|: "
            f"{diagnostics['pre_treatment_balance']['maximum_absolute_smd']:.4f}"
        ),
        f"- A/A empirical false-positive rate: {aa_text}",
        "",
        "## Randomized A/B experiment",
        "",
        "| Metric | Role | Absolute effect | 95% CI | p-value |",
        "|---|---|---:|---:|---:|",
    ]
    roles = {
        "next_day_retention": "primary",
        "clicked": "secondary",
        "watch_time_minutes": "secondary",
        "negative_feedback": "guardrail",
    }
    for name, role in roles.items():
        value = randomized["metrics"][name]
        lines.append(
            f"| {name} | {role} | {value['absolute_effect']:.4f} | "
            f"[{value['confidence_interval_lower']:.4f}, "
            f"{value['confidence_interval_upper']:.4f}] | {value['raw_p_value']:.4g} |"
        )
    cuped = randomized["cuped"]
    lines += [
        "",
        f"- Guardrail status: `{randomized['metrics']['negative_feedback']['guardrail_status']}`",
        f"- CUPED variance reduction: {cuped.get('variance_reduction_percent', 'skipped')}",
        "- The only predefined subgroup is `is_new_user`; interaction tests use HC3.",
        "",
        "## Observational causal analysis",
        "",
        "| Metric | Naive difference | Untrimmed stabilized IPTW ATE | Trimmed sensitivity |",
        "|---|---:|---:|---:|",
    ]
    for name, value in observational["effects"].items():
        lines.append(
            f"| {name} | {value['naive_effect']:.4f} | {value['iptw_ate']:.4f} | "
            f"{value['trimmed_iptw_ate']:.4f} |"
        )
    lines += [
        "",
        (
            f"- Propensity range: {observational['overlap']['overall']['min']:.4f} to "
            f"{observational['overlap']['overall']['max']:.4f}"
        ),
        f"- Overall ESS: {observational['weights']['effective_sample_size']['overall']:.1f}",
        (
            "- Maximum |SMD| before / after IPTW: "
            f"{observational['balance']['before']['maximum_absolute_smd']:.4f} / "
            f"{observational['balance']['iptw']['maximum_absolute_smd']:.4f}"
        ),
        "",
        "## Boundaries, risks, and open questions",
        "",
        "- All data and results are synthetic and cannot be generalized to real users.",
        "- IPTW adjusts measured covariates only and cannot remove unmeasured confounding.",
        "- Statistical significance does not replace effect size, guardrail, or domain review.",
        "- Independent synthetic validation exists but is not read by this formal report.",
        "- No automated launch decision is made; human review is required.",
        "",
        "## Reproduce",
        "",
        "```bash",
        (
            "python -m growth_experiment_lab run-all --config "
            f"configs/{config['portfolio']['config_label']}.toml --overwrite"
        ),
        "```",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_examples(config: dict[str, Any], overwrite: bool = False) -> list[Path]:
    """Export only approved formal reports and plots; never copy truth validation."""
    if config["portfolio"]["config_label"] != "demo":
        raise PortfolioError("Public examples must be generated with configs/demo.toml")
    destination = Path(config["portfolio"]["examples_dir"])
    mapping = {
        config["portfolio"]["report_path"]: "portfolio_report.md",
        config["ab_analysis"]["markdown_path"]: "ab_analysis.md",
        config["observational"]["markdown_path"]: "observational_analysis.md",
        config["randomization_diagnostics"]["balance_plot_path"]: "randomization_balance.png",
        config["ab_analysis"]["effect_plot_path"]: "ab_effect_forest.png",
        config["observational"]["overlap_plot_path"]: "propensity_overlap.png",
        config["observational"]["balance_plot_path"]: "iptw_balance_love_plot.png",
    }
    outputs = [destination / name for name in mapping.values()]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise PortfolioError(
            f"Example files already exist: {', '.join(existing)}. Use --overwrite."
        )
    destination.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in mapping.items():
        source = Path(source_name)
        if not source.is_file():
            raise PortfolioError(f"Required formal artifact does not exist: {source}")
        output = destination / output_name
        if output.suffix == ".md":
            body = source.read_text(encoding="utf-8")
            notice = "> Generated from synthetic data with `configs/demo.toml`.\n\n"
            output.write_text(notice + body, encoding="utf-8")
        else:
            image = plt.imread(source)
            figure, axis = plt.subplots(figsize=(10, 6))
            axis.imshow(image)
            axis.axis("off")
            figure.text(
                0.5,
                0.01,
                "Generated from synthetic data with configs/demo.toml",
                ha="center",
                fontsize=9,
            )
            figure.tight_layout(rect=(0, 0.03, 1, 1))
            figure.savefig(output, dpi=120)
            plt.close(figure)
    return outputs
