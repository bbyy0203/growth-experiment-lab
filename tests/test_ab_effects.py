"""Tests for randomized binary and Welch continuous ITT estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from growth_experiment_lab.ab_effects import (
    analyze_binary_metric,
    analyze_continuous_metric,
    apply_secondary_holm,
    guardrail_status,
)


def test_binary_known_effect_relative_lift_and_statsmodels_pvalue() -> None:
    frame = pd.DataFrame(
        {"treatment": [0] * 100 + [1] * 100, "outcome": [1] * 20 + [0] * 80 + [1] * 30 + [0] * 70}
    )
    result = analyze_binary_metric(frame, "outcome")
    _, expected_p = proportions_ztest([30, 20], [100, 100])

    assert result["absolute_effect"] == pytest.approx(0.10)
    assert result["relative_lift"] == pytest.approx(0.50)
    assert result["raw_p_value"] == pytest.approx(expected_p)
    assert 0 <= result["raw_p_value"] <= 1
    assert result["confidence_interval_lower"] < result["confidence_interval_upper"]


def test_binary_zero_control_rate_has_no_relative_lift() -> None:
    frame = pd.DataFrame({"treatment": [0, 0, 1, 1], "outcome": [0, 0, 0, 1]})
    result = analyze_binary_metric(frame, "outcome")

    assert result["relative_lift"] is None
    assert result["relative_lift_interpretable"] is False


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [(0.001, 0.004, "pass"), (0.006, 0.010, "fail"), (0.001, 0.008, "inconclusive")],
)
def test_guardrail_three_way_status(lower: float, upper: float, expected: str) -> None:
    result = {"confidence_interval_lower": lower, "confidence_interval_upper": upper}
    assert guardrail_status(result, 0.005) == expected


def test_welch_estimator_matches_scipy_and_keeps_zero_users() -> None:
    control = np.array([0.0, 0.0, 1.0, 2.0, 10.0])
    treatment = np.array([0.0, 1.0, 2.0, 8.0, 20.0, 30.0])
    frame = pd.DataFrame(
        {
            "treatment": [0] * len(control) + [1] * len(treatment),
            "watch": np.concatenate([control, treatment]),
        }
    )
    result = analyze_continuous_metric(frame, "watch")
    expected = stats.ttest_ind(treatment, control, equal_var=False)

    assert result["control_n"] == len(control)
    assert result["treatment_n"] == len(treatment)
    assert result["t_statistic"] == pytest.approx(expected.statistic)
    assert result["raw_p_value"] == pytest.approx(expected.pvalue)
    assert result["standard_error"] > 0
    assert result["welch_degrees_of_freedom"] > 0
    assert result["confidence_interval_lower"] < result["confidence_interval_upper"]


def test_extremely_skewed_continuous_values_do_not_crash() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [0] * 50 + [1] * 50,
            "watch": [0.0] * 49 + [10_000.0] + [0.0] * 48 + [1.0, 20_000.0],
        }
    )
    result = analyze_continuous_metric(frame, "watch")
    assert np.isfinite(result["raw_p_value"])


def test_holm_known_values_and_declared_family_only() -> None:
    metrics = {
        "next_day_retention": {"raw_p_value": 0.001, "adjusted_p_value": None},
        "clicked": {"raw_p_value": 0.01, "adjusted_p_value": None},
        "watch_time_minutes": {"raw_p_value": 0.04, "adjusted_p_value": None},
        "negative_feedback": {"raw_p_value": 0.02, "adjusted_p_value": None},
    }
    expected = multipletests([0.01, 0.04], alpha=0.05, method="holm")[1]
    apply_secondary_holm(metrics, 0.05)

    assert metrics["clicked"]["adjusted_p_value"] == pytest.approx(expected[0])
    assert metrics["watch_time_minutes"]["adjusted_p_value"] == pytest.approx(expected[1])
    assert metrics["next_day_retention"]["adjusted_p_value"] is None
    assert metrics["negative_feedback"]["adjusted_p_value"] is None
    assert metrics["clicked"]["adjusted_p_value"] <= metrics["watch_time_minutes"][
        "adjusted_p_value"
    ]
