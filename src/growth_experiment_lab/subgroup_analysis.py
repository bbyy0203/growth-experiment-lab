"""Predeclared is_new_user subgroup effects and formal interaction tests."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import statsmodels.formula.api as smf

from growth_experiment_lab.ab_effects import (
    analyze_binary_metric,
    analyze_continuous_metric,
)


class SubgroupAnalysisError(ValueError):
    """Raised when subgroup analysis deviates from the predeclared contract."""


def analyze_predefined_subgroups(
    frame: pd.DataFrame,
    metric_name: str,
    metric_type: Literal["binary", "continuous"],
    alpha: float,
    confidence_level: float,
    subgroup: str = "is_new_user",
) -> dict[str, Any]:
    """Estimate both subgroup effects and an HC3 treatment-by-subgroup interaction."""
    if subgroup != "is_new_user":
        raise SubgroupAnalysisError("only the predeclared is_new_user subgroup is allowed")
    if not bool(frame[subgroup].isin([0, 1]).all()):
        raise SubgroupAnalysisError("is_new_user must contain only 0/1")
    analyzer = analyze_binary_metric if metric_type == "binary" else analyze_continuous_metric
    subgroup_results: dict[str, Any] = {}
    for value, label in ((0, "existing_users"), (1, "new_users")):
        subset = frame.loc[frame[subgroup].eq(value)]
        result = analyzer(subset, metric_name, confidence_level)
        subgroup_results[label] = {
            "is_new_user": value,
            "total_n": len(subset),
            "control_n": result["control_n"],
            "treatment_n": result["treatment_n"],
            "absolute_effect": result["absolute_effect"],
            "confidence_interval_lower": result["confidence_interval_lower"],
            "confidence_interval_upper": result["confidence_interval_upper"],
            "raw_p_value": result["raw_p_value"],
        }

    model_frame = frame[[metric_name, "treatment", subgroup]].astype(float)
    model = smf.ols(
        f"{metric_name} ~ treatment + {subgroup} + treatment:{subgroup}",
        data=model_frame,
    ).fit(cov_type="HC3")
    interaction_name = f"treatment:{subgroup}"
    interaction_ci = model.conf_int(alpha=1.0 - confidence_level).loc[interaction_name]
    interaction_p_value = float(model.pvalues[interaction_name])
    return {
        "metric_name": metric_name,
        "predefined_subgroup": subgroup,
        "subgroup_effects": subgroup_results,
        "interaction": {
            "coefficient": float(model.params[interaction_name]),
            "standard_error": float(model.bse[interaction_name]),
            "p_value": interaction_p_value,
            "confidence_level": confidence_level,
            "confidence_interval_lower": float(interaction_ci.iloc[0]),
            "confidence_interval_upper": float(interaction_ci.iloc[1]),
            "robust_covariance": "HC3",
            "evidence_of_heterogeneity": bool(interaction_p_value < alpha),
        },
        "interpretation_note": (
            "Heterogeneity is assessed by the interaction, not by comparing subgroup p-values."
        ),
        "multiplicity_note": (
            "Predefined supplemental analysis; not in the primary/secondary family."
        ),
    }
