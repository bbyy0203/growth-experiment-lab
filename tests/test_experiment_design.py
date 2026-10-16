"""Tests for prospective sample-size and MDE calculations."""

from __future__ import annotations

import pytest

from growth_experiment_lab.experiment_design import (
    ExperimentDesignError,
    binary_mde,
    binary_sample_size,
    continuous_sample_size,
)


def _plan(**overrides: float) -> dict[str, object]:
    arguments = {
        "baseline_probability": 0.30,
        "absolute_effect": 0.01,
        "alpha": 0.05,
        "power": 0.80,
        "treatment_share": 0.50,
    }
    arguments.update(overrides)
    return binary_sample_size(**arguments)


def test_binary_sample_size_returns_positive_integer_groups() -> None:
    result = _plan()

    assert isinstance(result["control_n"], int) and result["control_n"] > 0
    assert isinstance(result["treatment_n"], int) and result["treatment_n"] > 0
    assert result["total_n"] == result["control_n"] + result["treatment_n"]


def test_smaller_mde_requires_more_users() -> None:
    assert _plan(absolute_effect=0.005)["total_n"] > _plan(absolute_effect=0.02)["total_n"]


def test_higher_power_requires_more_users() -> None:
    assert _plan(power=0.90)["total_n"] > _plan(power=0.70)["total_n"]


def test_stricter_alpha_requires_more_users() -> None:
    assert _plan(alpha=0.01)["total_n"] > _plan(alpha=0.10)["total_n"]


def test_unbalanced_allocation_is_reflected_in_group_sizes() -> None:
    result = _plan(treatment_share=0.30)

    assert result["treatment_n"] < result["control_n"]
    assert result["treatment_n"] / result["control_n"] == pytest.approx(0.3 / 0.7, rel=0.01)


@pytest.mark.parametrize(
    ("baseline", "effect"),
    [(0.99, 0.02), (0.30, 0.0), (0.30, -0.01)],
)
def test_invalid_binary_effect_inputs_fail(baseline: float, effect: float) -> None:
    with pytest.raises(ExperimentDesignError):
        binary_sample_size(baseline, effect, 0.05, 0.8, 0.5)


def test_mde_approximately_inverts_sample_size() -> None:
    planned = _plan(absolute_effect=0.02)
    inverted = binary_mde(
        0.30,
        planned["control_n"],
        planned["treatment_n"],
        0.05,
        0.80,
    )

    assert inverted["detectable_absolute_effect"] == pytest.approx(0.02, abs=0.0001)


def test_mde_reports_infeasible_small_design() -> None:
    with pytest.raises(ExperimentDesignError, match="no positive MDE solution"):
        binary_mde(0.50, 1, 1, 0.05, 0.99)


@pytest.mark.parametrize("standard_deviation", [0.0, -1.0])
def test_continuous_design_rejects_invalid_standard_deviation(
    standard_deviation: float,
) -> None:
    with pytest.raises(ExperimentDesignError, match="pilot_standard_deviation"):
        continuous_sample_size(1.0, standard_deviation, 0.05, 0.8, 0.5)


def test_continuous_design_returns_standardized_effect() -> None:
    result = continuous_sample_size(2.0, 10.0, 0.05, 0.8, 0.5)

    assert result["standardized_effect_size"] == pytest.approx(0.2)
    assert result["total_n"] > 0
