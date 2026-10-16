"""Tests for stratified user-level Bootstrap sensitivity intervals."""

from __future__ import annotations

import pandas as pd

from growth_experiment_lab.bootstrap import bootstrap_mean_difference


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"u{index}" for index in range(20)],
            "treatment": [0] * 10 + [1] * 10,
            "binary": [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            "continuous": list(range(10)) + list(range(3, 13)),
        }
    )


def test_bootstrap_is_reproducible_and_seed_sensitive() -> None:
    first = bootstrap_mean_difference(_frame(), "continuous", "continuous", 100, 10)
    second = bootstrap_mean_difference(_frame(), "continuous", "continuous", 100, 10)
    changed = bootstrap_mean_difference(_frame(), "continuous", "continuous", 100, 11)

    assert first == second
    assert first != changed


def test_binary_and_continuous_bootstrap_fields() -> None:
    for metric, metric_type in (("binary", "binary"), ("continuous", "continuous")):
        result = bootstrap_mean_difference(_frame(), metric, metric_type, 50, 7)
        assert result["sampling_unit"] == "user_id"
        assert result["stratified_by_treatment"] is True
        assert result["successful_resamples"] == 50
        assert result["failed_resamples"] == 0
        assert result["percentile_confidence_interval_lower"] <= result[
            "percentile_confidence_interval_upper"
        ]


def test_group_stratification_preserves_constant_group_effect() -> None:
    frame = pd.DataFrame(
        {"user_id": range(8), "treatment": [0] * 4 + [1] * 4, "value": [0] * 4 + [1] * 4}
    )
    result = bootstrap_mean_difference(frame, "value", "binary", 30, 5)
    assert result["bootstrap_mean_effect"] == 1.0
    assert result["percentile_confidence_interval_lower"] == 1.0
    assert result["percentile_confidence_interval_upper"] == 1.0
