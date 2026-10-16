import numpy as np
import pandas as pd
import pytest

from growth_experiment_lab.iptw import IPTWError, balance_table, hajek_effect


def test_hajek_known_example():
    data = pd.DataFrame({"treatment": [0, 0, 1, 1], "clicked": [0, 1, 1, 1]})
    result = hajek_effect(data, "clicked", np.array([1.0, 1.0, 1.0, 2.0]))
    assert result["control_mean"] == 0.5 and result["ate"] == 0.5


def test_hajek_requires_groups():
    with pytest.raises(IPTWError, match="Both"):
        hajek_effect(pd.DataFrame({"treatment": [1], "clicked": [1]}), "clicked", np.ones(1))


def test_balance_expands_categories():
    data = pd.DataFrame(
        {
            "treatment": [0, 0, 1, 1],
            "is_new_user": [0, 1, 0, 1],
            "pre_7d_sessions": [1, 2, 1, 2],
            "pre_7d_watch_time_minutes": [1.0, 2.0, 1.0, 2.0],
            "pre_7d_click_days": [0, 1, 0, 1],
            "pre_experiment_activity_score": [0.0, 1.0, 0.0, 1.0],
            "device_type": ["a", "b", "a", "b"],
            "region": ["x", "y", "x", "y"],
        }
    )
    rows = balance_table(data)
    assert {r["covariate"] for r in rows} >= {"device_type=a", "region=y"}
    assert all(r["absolute_smd"] == 0 for r in rows)
