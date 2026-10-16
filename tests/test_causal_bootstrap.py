import numpy as np
import pandas as pd

from growth_experiment_lab.causal_bootstrap import bootstrap_iptw


def _data(n=180):
    rng = np.random.default_rng(19)
    activity = rng.normal(size=n)
    treatment = rng.binomial(1, 1 / (1 + np.exp(-activity)))
    return pd.DataFrame(
        {
            "treatment": treatment,
            "is_new_user": rng.binomial(1, 0.3, n),
            "pre_7d_sessions": rng.poisson(4, n),
            "pre_7d_watch_time_minutes": rng.gamma(2, 8, n),
            "pre_7d_click_days": rng.integers(0, 8, n),
            "pre_experiment_activity_score": activity,
            "device_type": rng.choice(["ios", "web"], n),
            "region": rng.choice(["north", "south"], n),
            "next_day_retention": rng.binomial(1, 0.3, n),
            "clicked": rng.binomial(1, 0.2, n),
            "watch_time_minutes": rng.gamma(2, 9, n),
            "negative_feedback": rng.binomial(1, 0.04, n),
        }
    )


def test_bootstrap_is_reproducible_and_reports_refit():
    data = _data()
    first = bootstrap_iptw(data, 8, 42)
    second = bootstrap_iptw(data, 8, 42)
    assert first == second
    assert first["clicked"]["refit_propensity_each_resample"] is True
    assert first["clicked"]["sampling_unit"] == "user"
    assert first["clicked"]["successful_resamples"] == 8


def test_bootstrap_changes_with_seed():
    assert bootstrap_iptw(_data(), 8, 42) != bootstrap_iptw(_data(), 8, 43)
