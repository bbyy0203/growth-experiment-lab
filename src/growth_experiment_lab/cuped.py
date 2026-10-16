"""CUPED variance reduction for the predeclared watch-time outcome."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from growth_experiment_lab.ab_effects import analyze_continuous_metric


class CupedError(ValueError):
    """Raised when CUPED cannot be applied under its pre-treatment contract."""


def analyze_watch_time_cuped(
    frame: pd.DataFrame,
    confidence_level: float = 0.95,
    outcome: str = "watch_time_minutes",
    covariate: str = "pre_7d_watch_time_minutes",
) -> dict[str, Any]:
    """Estimate one theta on all users and compare raw versus adjusted Welch inference."""
    if outcome != "watch_time_minutes":
        raise CupedError("CUPED v1 supports only watch_time_minutes")
    if covariate != "pre_7d_watch_time_minutes":
        raise CupedError("CUPED covariate must be pre_7d_watch_time_minutes")
    if frame[[outcome, covariate]].isna().any().any():
        raise CupedError("CUPED outcome and covariate must not contain missing values")
    x = frame[covariate].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    x_variance = float(np.var(x, ddof=1))
    if x_variance == 0:
        raise CupedError("CUPED covariate has zero variance")
    covariance = float(np.cov(y, x, ddof=1)[0, 1])
    theta = covariance / x_variance
    adjusted = y - theta * (x - x.mean())
    analysis_frame = frame[["treatment"]].copy()
    analysis_frame["watch_time_minutes_cuped"] = adjusted
    raw_result = analyze_continuous_metric(frame, outcome, confidence_level)
    adjusted_result = analyze_continuous_metric(
        analysis_frame, "watch_time_minutes_cuped", confidence_level
    )
    raw_variance = float(np.var(y, ddof=1))
    cuped_variance = float(np.var(adjusted, ddof=1))
    variance_reduction = (1.0 - cuped_variance / raw_variance) * 100.0
    control_mask = frame["treatment"].eq(0).to_numpy()
    treatment_mask = ~control_mask
    return {
        "outcome": outcome,
        "pretreatment_covariate": covariate,
        "theta": theta,
        "covariate_mean": float(x.mean()),
        "control_raw_mean": float(y[control_mask].mean()),
        "treatment_raw_mean": float(y[treatment_mask].mean()),
        "control_cuped_mean": float(adjusted[control_mask].mean()),
        "treatment_cuped_mean": float(adjusted[treatment_mask].mean()),
        "raw_absolute_effect": raw_result["absolute_effect"],
        "cuped_absolute_effect": adjusted_result["absolute_effect"],
        "raw_variance": raw_variance,
        "cuped_variance": cuped_variance,
        "variance_reduction_percent": variance_reduction,
        "raw_standard_error": raw_result["standard_error"],
        "cuped_standard_error": adjusted_result["standard_error"],
        "confidence_level": confidence_level,
        "confidence_interval_lower": adjusted_result["confidence_interval_lower"],
        "confidence_interval_upper": adjusted_result["confidence_interval_upper"],
        "p_value": adjusted_result["raw_p_value"],
        "method_note": (
            "One theta estimated on all randomized users using a pre-treatment covariate."
        ),
    }
