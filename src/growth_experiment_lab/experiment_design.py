"""Prospective sample-size and minimum-detectable-effect calculations."""

from __future__ import annotations

import math
from typing import Any

from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


class ExperimentDesignError(ValueError):
    """Raised when a requested prospective design is invalid or infeasible."""


def _validate_common(alpha: float, power: float, treatment_share: float, alternative: str) -> None:
    if not 0 < alpha < 1:
        raise ExperimentDesignError("alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ExperimentDesignError("power must be between 0 and 1")
    if not 0 < treatment_share < 1:
        raise ExperimentDesignError("treatment_share must be between 0 and 1")
    if alternative != "two-sided":
        raise ExperimentDesignError("only the 'two-sided' alternative is supported")


def binary_sample_size(
    baseline_probability: float,
    absolute_effect: float,
    alpha: float,
    power: float,
    treatment_share: float,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Plan two independent proportions using Cohen's h and normal power."""
    _validate_common(alpha, power, treatment_share, alternative)
    if not 0 < baseline_probability < 1:
        raise ExperimentDesignError("baseline_probability must be between 0 and 1")
    if absolute_effect <= 0:
        raise ExperimentDesignError("absolute_effect must be positive")
    treatment_probability = baseline_probability + absolute_effect
    if not 0 < treatment_probability < 1:
        raise ExperimentDesignError("implied treatment probability must be between 0 and 1")

    allocation_ratio = treatment_share / (1.0 - treatment_share)
    effect_size = abs(proportion_effectsize(treatment_probability, baseline_probability))
    control_n_float = NormalIndPower().solve_power(
        effect_size=effect_size,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=allocation_ratio,
        alternative=alternative,
    )
    control_n = math.ceil(control_n_float)
    treatment_n = math.ceil(control_n_float * allocation_ratio)
    return {
        "control_n": control_n,
        "treatment_n": treatment_n,
        "total_n": control_n + treatment_n,
        "baseline_probability": baseline_probability,
        "treatment_probability": treatment_probability,
        "absolute_effect": absolute_effect,
        "relative_lift": absolute_effect / baseline_probability,
        "alpha": alpha,
        "power": power,
        "treatment_share": treatment_share,
        "allocation_ratio_treatment_to_control": allocation_ratio,
        "alternative": alternative,
    }


def binary_mde(
    baseline_probability: float,
    control_n: int,
    treatment_n: int,
    alpha: float,
    power: float,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Invert normal-theory power with bisection to obtain a positive absolute MDE."""
    treatment_share = treatment_n / (control_n + treatment_n) if control_n + treatment_n else 0
    _validate_common(alpha, power, treatment_share, alternative)
    if not 0 < baseline_probability < 1:
        raise ExperimentDesignError("baseline_probability must be between 0 and 1")
    if not isinstance(control_n, int) or control_n <= 0:
        raise ExperimentDesignError("control_n must be a positive integer")
    if not isinstance(treatment_n, int) or treatment_n <= 0:
        raise ExperimentDesignError("treatment_n must be a positive integer")

    ratio = treatment_n / control_n
    solver = NormalIndPower()

    def achieved_power(effect: float) -> float:
        treatment_probability = baseline_probability + effect
        effect_size = abs(proportion_effectsize(treatment_probability, baseline_probability))
        return float(
            solver.power(
                effect_size,
                nobs1=control_n,
                alpha=alpha,
                ratio=ratio,
                alternative=alternative,
            )
        )

    lower = 0.0
    upper = 1.0 - baseline_probability - 1e-12
    if upper <= 0 or achieved_power(upper) < power:
        raise ExperimentDesignError(
            "no positive MDE solution exists within the probability boundary"
        )
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if achieved_power(midpoint) >= power:
            upper = midpoint
        else:
            lower = midpoint
    detectable_effect = upper
    return {
        "detectable_absolute_effect": detectable_effect,
        "detectable_relative_lift": detectable_effect / baseline_probability,
        "implied_treatment_probability": baseline_probability + detectable_effect,
        "baseline_probability": baseline_probability,
        "control_n": control_n,
        "treatment_n": treatment_n,
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
        "search_lower_bound": 0.0,
        "search_upper_bound": 1.0 - baseline_probability,
    }


def continuous_sample_size(
    expected_mean_difference: float,
    pilot_standard_deviation: float,
    alpha: float,
    power: float,
    treatment_share: float,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Plan a two-sample continuous outcome using a common-SD Welch approximation."""
    _validate_common(alpha, power, treatment_share, alternative)
    if expected_mean_difference == 0:
        raise ExperimentDesignError("expected_mean_difference must be non-zero")
    if pilot_standard_deviation <= 0:
        raise ExperimentDesignError("pilot_standard_deviation must be positive")
    standardized_effect = abs(expected_mean_difference) / pilot_standard_deviation
    ratio = treatment_share / (1.0 - treatment_share)
    control_float = TTestIndPower().solve_power(
        effect_size=standardized_effect,
        nobs1=None,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    control_n = math.ceil(control_float)
    treatment_n = math.ceil(control_float * ratio)
    return {
        "standardized_effect_size": standardized_effect,
        "control_n": control_n,
        "treatment_n": treatment_n,
        "total_n": control_n + treatment_n,
        "allocation_ratio_treatment_to_control": ratio,
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
        "method": "two-sample t-test approximation for Welch planning",
    }
