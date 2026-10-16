"""ITT estimators for binary and continuous outcomes in a randomized experiment."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest


class EffectEstimationError(ValueError):
    """Raised when an ITT effect cannot be estimated under the declared contract."""


def _split_groups(frame: pd.DataFrame, metric_name: str) -> tuple[pd.Series, pd.Series]:
    if metric_name not in frame.columns:
        raise EffectEstimationError(f"metric is missing: {metric_name}")
    if frame[metric_name].isna().any():
        raise EffectEstimationError(f"metric contains missing values: {metric_name}")
    control = frame.loc[frame["treatment"].eq(0), metric_name]
    treatment = frame.loc[frame["treatment"].eq(1), metric_name]
    if control.empty or treatment.empty:
        raise EffectEstimationError("both treatment groups must contain users")
    return control, treatment


def analyze_binary_metric(
    frame: pd.DataFrame,
    metric_name: str,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate a binary risk difference with z-test and Newcombe-Wilson interval."""
    control, treatment = _split_groups(frame, metric_name)
    if not bool(control.isin([0, 1]).all()) or not bool(treatment.isin([0, 1]).all()):
        raise EffectEstimationError(f"binary metric must contain only 0/1: {metric_name}")
    control_n = len(control)
    treatment_n = len(treatment)
    control_events = int(control.sum())
    treatment_events = int(treatment.sum())
    control_rate = control_events / control_n
    treatment_rate = treatment_events / treatment_n
    effect = treatment_rate - control_rate
    relative_lift = None if control_rate == 0 else effect / control_rate
    standard_error = math.sqrt(
        control_rate * (1 - control_rate) / control_n
        + treatment_rate * (1 - treatment_rate) / treatment_n
    )
    z_statistic, p_value = proportions_ztest(
        [treatment_events, control_events], [treatment_n, control_n], alternative="two-sided"
    )
    alpha = 1.0 - confidence_level
    ci_lower, ci_upper = confint_proportions_2indep(
        treatment_events,
        treatment_n,
        control_events,
        control_n,
        method="newcomb",
        compare="diff",
        alpha=alpha,
    )
    return {
        "metric_name": metric_name,
        "metric_type": "binary",
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_events": control_events,
        "treatment_events": treatment_events,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_effect": effect,
        "relative_lift": relative_lift,
        "relative_lift_interpretable": relative_lift is not None,
        "standard_error": standard_error,
        "z_statistic": float(z_statistic),
        "raw_p_value": float(p_value),
        "confidence_level": confidence_level,
        "confidence_interval_lower": float(ci_lower),
        "confidence_interval_upper": float(ci_upper),
        "adjusted_p_value": None,
        "statistical_significance": bool(p_value < alpha),
        "confidence_interval_method": "Newcombe-Wilson score interval for independent proportions",
        "business_interpretation": _effect_interpretation(effect, p_value, alpha),
    }


def analyze_continuous_metric(
    frame: pd.DataFrame,
    metric_name: str,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate a mean difference with Welch's unequal-variance t procedure."""
    control, treatment = _split_groups(frame, metric_name)
    control = control.astype(float)
    treatment = treatment.astype(float)
    control_n = len(control)
    treatment_n = len(treatment)
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    control_variance = float(control.var(ddof=1))
    treatment_variance = float(treatment.var(ddof=1))
    standard_error = math.sqrt(
        control_variance / control_n + treatment_variance / treatment_n
    )
    if standard_error == 0:
        raise EffectEstimationError(f"zero standard error for continuous metric: {metric_name}")
    numerator = (control_variance / control_n + treatment_variance / treatment_n) ** 2
    denominator = (control_variance / control_n) ** 2 / (control_n - 1) + (
        treatment_variance / treatment_n
    ) ** 2 / (treatment_n - 1)
    degrees_of_freedom = numerator / denominator
    test = stats.ttest_ind(treatment, control, equal_var=False)
    effect = treatment_mean - control_mean
    alpha = 1.0 - confidence_level
    critical_value = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    margin = critical_value * standard_error
    relative_lift = None if control_mean == 0 else effect / control_mean
    return {
        "metric_name": metric_name,
        "metric_type": "continuous",
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "control_std": math.sqrt(control_variance),
        "treatment_std": math.sqrt(treatment_variance),
        "control_median": float(control.median()),
        "treatment_median": float(treatment.median()),
        "control_p95": float(control.quantile(0.95)),
        "treatment_p95": float(treatment.quantile(0.95)),
        "absolute_effect": effect,
        "relative_lift": relative_lift,
        "relative_lift_interpretable": relative_lift is not None,
        "standard_error": standard_error,
        "welch_degrees_of_freedom": degrees_of_freedom,
        "t_statistic": float(test.statistic),
        "raw_p_value": float(test.pvalue),
        "confidence_level": confidence_level,
        "confidence_interval_lower": effect - margin,
        "confidence_interval_upper": effect + margin,
        "adjusted_p_value": None,
        "statistical_significance": bool(test.pvalue < alpha),
        "confidence_interval_method": "Welch-Satterthwaite unequal-variance t interval",
        "business_interpretation": _effect_interpretation(effect, float(test.pvalue), alpha),
    }


def apply_secondary_holm(
    metric_results: dict[str, dict[str, Any]], alpha: float
) -> dict[str, dict[str, Any]]:
    """Apply the predeclared Holm correction to clicked and watch time only."""
    family = ["clicked", "watch_time_minutes"]
    if any(metric not in metric_results for metric in family):
        raise EffectEstimationError("Holm family requires clicked and watch_time_minutes")
    raw = [metric_results[metric]["raw_p_value"] for metric in family]
    rejected, adjusted, _, _ = multipletests(raw, alpha=alpha, method="holm")
    for metric, adjusted_p_value, is_rejected in zip(family, adjusted, rejected, strict=True):
        metric_results[metric]["adjusted_p_value"] = float(adjusted_p_value)
        metric_results[metric]["statistical_significance"] = bool(is_rejected)
    return metric_results


def guardrail_status(result: dict[str, Any], maximum_allowed_increase: float) -> str:
    """Classify a harmful-positive guardrail effect against its business threshold."""
    lower = result["confidence_interval_lower"]
    upper = result["confidence_interval_upper"]
    if upper <= maximum_allowed_increase:
        return "pass"
    if lower > maximum_allowed_increase:
        return "fail"
    return "inconclusive"


def _effect_interpretation(effect: float, p_value: float, alpha: float) -> str:
    direction = "positive" if effect > 0 else "negative" if effect < 0 else "zero"
    evidence = "statistically significant" if p_value < alpha else "statistically inconclusive"
    return f"Observed ITT effect is {direction} and {evidence}; business importance is separate."
