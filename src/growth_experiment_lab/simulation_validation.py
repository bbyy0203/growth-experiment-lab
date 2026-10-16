"""Isolated synthetic-truth validation, intentionally separate from formal estimation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SimulationValidationError(RuntimeError):
    """Raised when isolated synthetic-estimator validation cannot run safely."""


def validate_simulation_estimates(
    config: dict[str, Any], overwrite: bool = False
) -> dict[str, Any]:
    """Compare completed formal estimates with synthetic randomized sample ATEs."""
    results_path = Path(config["ab_analysis"]["results_path"])
    truth_path = Path(config["data_generation"]["output_dir"]) / "metadata" / "true_effects.json"
    output_path = Path(config["ab_analysis"]["simulation_validation_path"])
    if not results_path.is_file():
        raise SimulationValidationError(f"Formal A/B results do not exist: {results_path}")
    if not truth_path.is_file():
        raise SimulationValidationError(f"Synthetic truth metadata does not exist: {truth_path}")
    if output_path.exists() and not overwrite:
        raise SimulationValidationError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    formal = json.loads(results_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    true_effects = truth["true_sample_ate"]["randomized"]
    comparisons = {}
    for metric_name, true_effect in true_effects.items():
        estimate = formal["metrics"][metric_name]
        error = estimate["absolute_effect"] - true_effect
        comparisons[metric_name] = {
            "estimated_effect": estimate["absolute_effect"],
            "true_sample_ate": true_effect,
            "estimation_error": error,
            "absolute_error": abs(error),
            "true_ate_in_reported_95_percent_ci": bool(
                estimate["confidence_interval_lower"]
                <= true_effect
                <= estimate["confidence_interval_upper"]
            ),
        }
    report = {
        "synthetic_validation_only": True,
        "not_for_business_decision_support": True,
        "formal_results_path": str(results_path),
        "synthetic_metadata_path": str(truth_path),
        "comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
