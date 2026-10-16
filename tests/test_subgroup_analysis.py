"""Tests for predeclared subgroup effects and HC3 interaction inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growth_experiment_lab.subgroup_analysis import (
    SubgroupAnalysisError,
    analyze_predefined_subgroups,
)


def _interaction_frame() -> pd.DataFrame:
    rng = np.random.default_rng(55)
    users = 4_000
    treatment = rng.binomial(1, 0.5, users)
    is_new = rng.binomial(1, 0.4, users)
    outcome = 10 + treatment * 1.0 + is_new * 2.0 + treatment * is_new * 3.0
    outcome = outcome + rng.normal(0, 1, users)
    return pd.DataFrame(
        {"treatment": treatment, "is_new_user": is_new, "outcome": outcome}
    )


def test_subgroups_and_hc3_interaction_are_reported() -> None:
    result = analyze_predefined_subgroups(
        _interaction_frame(), "outcome", "continuous", 0.05, 0.95
    )

    assert set(result["subgroup_effects"]) == {"existing_users", "new_users"}
    assert result["interaction"]["robust_covariance"] == "HC3"
    assert result["interaction"]["coefficient"] == pytest.approx(3.0, abs=0.15)
    assert result["interaction"]["evidence_of_heterogeneity"] is True
    assert "interaction" in result["interpretation_note"].lower()


@pytest.mark.parametrize("subgroup", ["region", "device_type", "clicked"])
def test_automatic_or_outcome_defined_subgroups_are_rejected(subgroup: str) -> None:
    with pytest.raises(SubgroupAnalysisError, match="is_new_user"):
        analyze_predefined_subgroups(
            _interaction_frame(), "outcome", "continuous", 0.05, 0.95, subgroup=subgroup
        )
