"""User-level bootstrap that refits the propensity model in every resample."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from growth_experiment_lab.iptw import hajek_effect
from growth_experiment_lab.propensity import OUTCOMES, fit_propensity, stabilized_weights


class CausalBootstrapError(RuntimeError):
    """Raised when too few causal bootstrap replications succeed."""


def bootstrap_iptw(frame: pd.DataFrame, resamples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = {metric: [] for metric in OUTCOMES}
    failures = 0
    for _ in range(resamples):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))].reset_index(drop=True)
        try:
            scores, _ = fit_propensity(sample)
            weights = stabilized_weights(sample["treatment"], scores)
            for metric in OUTCOMES:
                estimates[metric].append(hajek_effect(sample, metric, weights)["ate"])
        except (ValueError, RuntimeError):
            failures += 1
    succeeded = resamples - failures
    if succeeded / resamples < 0.95:
        raise CausalBootstrapError(f"Bootstrap success rate below 95%: {succeeded}/{resamples}")
    result: dict[str, Any] = {}
    for metric, values in estimates.items():
        array = np.asarray(values)
        result[metric] = {
            "mean": float(array.mean()),
            "standard_error": float(array.std(ddof=1)),
            "confidence_interval_lower": float(np.quantile(array, 0.025)),
            "confidence_interval_upper": float(np.quantile(array, 0.975)),
            "successful_resamples": succeeded,
            "failed_resamples": failures,
            "success_rate": succeeded / resamples,
            "random_seed": seed,
            "refit_propensity_each_resample": True,
            "sampling_unit": "user",
        }
    return result
