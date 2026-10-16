"""Isolated synthetic truth validation for observational estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ObservationalValidationError(RuntimeError):
    """Raised when isolated observational validation cannot run."""


def validate_observational_simulation(
    config: dict[str, Any], overwrite: bool = False
) -> dict[str, Any]:
    settings = config["observational"]
    formal_path = Path(settings["results_path"])
    truth_path = Path(config["data_generation"]["output_dir"]) / "metadata" / "true_effects.json"
    output = Path(settings["simulation_validation_path"])
    if not formal_path.is_file():
        raise ObservationalValidationError(
            f"Formal observational results do not exist: {formal_path}"
        )
    if output.exists() and not overwrite:
        raise ObservationalValidationError(
            f"Output file already exists: {output}. Use --overwrite."
        )
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["true_sample_ate"]["observational"]
    comparisons = {}
    improved, not_improved = [], []
    for metric, true_ate in truth.items():
        estimate = formal["effects"][metric]
        naive_error = abs(estimate["naive_effect"] - true_ate)
        iptw_error = abs(estimate["iptw_ate"] - true_ate)
        bootstrap = estimate["bootstrap"]
        covered = None if bootstrap is None else bool(
            bootstrap["confidence_interval_lower"]
            <= true_ate
            <= bootstrap["confidence_interval_upper"]
        )
        comparison = {
            "naive_effect": estimate["naive_effect"],
            "iptw_estimate": estimate["iptw_ate"],
            "trimmed_iptw_estimate": estimate["trimmed_iptw_ate"],
            "true_sample_ate": true_ate,
            "naive_absolute_error": naive_error,
            "iptw_absolute_error": iptw_error,
            "trimmed_absolute_error": abs(estimate["trimmed_iptw_ate"] - true_ate),
            "true_ate_in_iptw_95_percent_ci": covered,
        }
        comparisons[metric] = comparison
        (improved if iptw_error < naive_error else not_improved).append(metric)
    report = {
        "synthetic_validation_only": True,
        "not_for_business_decision_support": True,
        "comparisons": comparisons,
        "summary": {
            "iptw_closer_than_naive_count": len(improved),
            "improved_metrics": improved,
            "not_improved_metrics": not_improved,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
