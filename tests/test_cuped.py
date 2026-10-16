"""Tests for pre-treatment-only CUPED variance reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growth_experiment_lab.cuped import CupedError, analyze_watch_time_cuped


def test_cuped_uses_one_full_sample_theta_and_reduces_variance() -> None:
    rng = np.random.default_rng(123)
    users = 5_000
    treatment = rng.binomial(1, 0.5, users)
    pre = rng.normal(20, 5, users)
    outcome = 5 + 1.8 * pre + 2.0 * treatment + rng.normal(0, 2, users)
    frame = pd.DataFrame(
        {
            "treatment": treatment,
            "pre_7d_watch_time_minutes": pre,
            "watch_time_minutes": outcome,
        }
    )
    result = analyze_watch_time_cuped(frame)
    expected_theta = np.cov(outcome, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)

    assert result["theta"] == pytest.approx(expected_theta)
    assert result["cuped_variance"] < result["raw_variance"]
    assert result["variance_reduction_percent"] > 0
    assert result["cuped_standard_error"] < result["raw_standard_error"]
    assert result["cuped_absolute_effect"] > 0


def test_cuped_rejects_zero_variance_covariate() -> None:
    frame = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "pre_7d_watch_time_minutes": [1.0] * 4,
            "watch_time_minutes": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(CupedError, match="zero variance"):
        analyze_watch_time_cuped(frame)


def test_cuped_rejects_unapproved_covariate() -> None:
    frame = pd.DataFrame(
        {"treatment": [0, 1], "watch_time_minutes": [1.0, 2.0], "clicked": [0, 1]}
    )
    with pytest.raises(CupedError, match="pre_7d_watch_time_minutes"):
        analyze_watch_time_cuped(frame, covariate="clicked")
