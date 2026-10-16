"""Tests for SRM and pre-treatment standardized mean differences."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from growth_experiment_lab.randomization_diagnostics import (
    RandomizationDiagnosticError,
    balance_diagnostics,
    binary_smd,
    continuous_smd,
    srm_check,
)


def test_equal_counts_pass_srm_and_fields_are_complete() -> None:
    result = srm_check(5_000, 5_000, 0.5, 0.001)

    assert result["passed"] is True
    assert set(result) == {
        "observed_control_count",
        "observed_treatment_count",
        "observed_treatment_share",
        "expected_control_count",
        "expected_treatment_count",
        "expected_treatment_share",
        "chi_square_statistic",
        "p_value",
        "srm_alpha",
        "passed",
        "interpretation",
    }


def test_obvious_sixty_forty_split_fails_srm() -> None:
    assert srm_check(6_000, 4_000, 0.5, 0.001)["passed"] is False


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1])
def test_invalid_expected_share_fails(share: float) -> None:
    with pytest.raises(RandomizationDiagnosticError, match="expected_treatment_share"):
        srm_check(50, 50, share, 0.001)


def test_srm_rejects_observational_data() -> None:
    with pytest.raises(RandomizationDiagnosticError, match="randomized"):
        srm_check(50, 50, 0.5, 0.001, dataset_type="observational")


def test_continuous_smd_known_example() -> None:
    control = pd.Series([0.0, 1.0, 2.0])
    treatment = pd.Series([1.0, 2.0, 3.0])
    result = continuous_smd(control, treatment)

    assert result["pooled_standard_deviation"] == pytest.approx(1.0)
    assert result["smd"] == pytest.approx(1.0)


def test_binary_smd_known_example() -> None:
    control = pd.Series([0, 0, 1, 1])
    treatment = pd.Series([0, 1, 1, 1])
    result = binary_smd(control, treatment)
    expected_sd = math.sqrt((0.5 * 0.5 + 0.75 * 0.25) / 2)

    assert result["smd"] == pytest.approx(0.25 / expected_sd)


def test_equal_constants_have_zero_smd() -> None:
    result = continuous_smd(pd.Series([2.0, 2.0]), pd.Series([2.0, 2.0]))
    assert result["smd"] == 0
    assert result["status"] == "constant_equal"


def test_different_zero_variance_groups_are_marked_infinite() -> None:
    result = continuous_smd(pd.Series([1.0, 1.0]), pd.Series([2.0, 2.0]))
    assert math.isinf(result["smd"])
    assert result["status"] == "zero_variance_mean_difference"


def _balance_mart() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_type": ["randomized"] * 8,
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
            "device_type": ["web", "web", "ios", "ios", "web", "ios", "ios", "ios"],
            "region": ["east", "west", "east", "west", "east", "east", "west", "west"],
            "is_new_user": [0, 0, 1, 1, 0, 1, 1, 1],
            "pre_7d_sessions": [1, 2, 3, 4, 2, 3, 4, 5],
            "pre_7d_watch_time_minutes": [1, 2, 3, 4, 2, 3, 4, 5],
            "pre_7d_click_days": [0, 1, 1, 2, 1, 2, 2, 3],
            "pre_experiment_activity_score": [-1, -0.5, 0.5, 1, -0.5, 0.5, 1, 1.5],
        }
    )


def test_categorical_balance_is_computed_per_level_and_warns() -> None:
    result = balance_diagnostics(_balance_mart(), warning_threshold=0.10)
    labels = {entry["label"] for entry in result["entries"]}

    assert {"device_type=ios", "device_type=web"}.issubset(labels)
    assert "device_type" in result["variable_maximum_absolute_smd"]
    assert result["has_warning"] is True
    assert result["warning_variables"]


@pytest.mark.parametrize(
    "forbidden", ["clicked", "watch_time_minutes", "assignment_probability"]
)
def test_balance_rejects_non_pretreatment_fields(forbidden: str) -> None:
    mart = _balance_mart()
    mart[forbidden] = 0

    with pytest.raises(RandomizationDiagnosticError, match="whitelist"):
        balance_diagnostics(mart, 0.10, covariates=[forbidden])


def test_default_balance_entries_exclude_outcomes() -> None:
    result = balance_diagnostics(_balance_mart(), warning_threshold=0.10)
    variables = {entry["variable"] for entry in result["entries"]}

    assert variables.isdisjoint(
        {"clicked", "watch_time_minutes", "next_day_retention", "negative_feedback"}
    )
