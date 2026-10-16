"""Stratified user-level bootstrap sensitivity intervals."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


class BootstrapError(ValueError):
    """Raised when a user-level bootstrap request is invalid."""


def bootstrap_mean_difference(
    frame: pd.DataFrame,
    metric_name: str,
    metric_type: Literal["binary", "continuous"],
    resamples: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Resample user rows with replacement separately within randomized groups."""
    if not isinstance(resamples, int) or resamples <= 0:
        raise BootstrapError("resamples must be a positive integer")
    control = frame.loc[frame["treatment"].eq(0), metric_name].to_numpy(dtype=float)
    treatment = frame.loc[frame["treatment"].eq(1), metric_name].to_numpy(dtype=float)
    if len(control) == 0 or len(treatment) == 0:
        raise BootstrapError("both treatment groups are required")
    if metric_type == "binary" and (
        not np.isin(control, [0, 1]).all() or not np.isin(treatment, [0, 1]).all()
    ):
        raise BootstrapError("binary bootstrap metric must contain only 0/1")
    if metric_type not in {"binary", "continuous"}:
        raise BootstrapError(f"unsupported metric_type: {metric_type}")

    rng = np.random.default_rng(random_seed)
    effects = np.empty(resamples, dtype=float)
    failed_resamples = 0
    for index in range(resamples):
        control_sample = control[rng.integers(0, len(control), len(control))]
        treatment_sample = treatment[rng.integers(0, len(treatment), len(treatment))]
        effect = treatment_sample.mean() - control_sample.mean()
        if np.isfinite(effect):
            effects[index - failed_resamples] = effect
        else:
            failed_resamples += 1
    successful = resamples - failed_resamples
    if successful == 0:
        raise BootstrapError("all bootstrap resamples failed")
    effects = effects[:successful]
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(effects, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "metric_name": metric_name,
        "metric_type": metric_type,
        "sampling_unit": "user_id",
        "stratified_by_treatment": True,
        "bootstrap_mean_effect": float(effects.mean()),
        "bootstrap_standard_error": float(effects.std(ddof=1)),
        "confidence_level": confidence_level,
        "percentile_confidence_interval_lower": float(lower),
        "percentile_confidence_interval_upper": float(upper),
        "requested_resamples": resamples,
        "successful_resamples": successful,
        "failed_resamples": failed_resamples,
        "random_seed": random_seed,
    }
