import numpy as np
import pandas as pd
import pytest

from growth_experiment_lab.propensity import (
    PROPENSITY_FEATURES,
    PropensityError,
    effective_sample_size,
    fit_propensity,
    stabilized_weights,
    trim_weights,
)


def frame(n=300):
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "treatment": rng.binomial(1, 0.5, n),
            "is_new_user": rng.binomial(1, 0.3, n),
            "pre_7d_sessions": rng.poisson(4, n),
            "pre_7d_watch_time_minutes": rng.gamma(2, 10, n),
            "pre_7d_click_days": rng.integers(0, 8, n),
            "pre_experiment_activity_score": rng.normal(size=n),
            "device_type": rng.choice(["ios", "android"], n),
            "region": rng.choice(["east", "west"], n),
        }
    )


def test_propensity_contract_and_reproducibility():
    data = frame()
    a, meta = fit_propensity(data)
    b, _ = fit_propensity(data)
    assert meta["features"] == PROPENSITY_FEATURES
    assert np.all((a > 0) & (a < 1))
    np.testing.assert_allclose(a, b)
    assert any("device_type" in x for x in meta["design_matrix_columns"])


def test_forbidden_feature_rejected():
    from growth_experiment_lab.propensity import validate_feature_contract

    with pytest.raises(PropensityError, match="Forbidden"):
        validate_feature_contract(frame(), ["clicked"])


def test_weights_ess_and_trim():
    treatment = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    weights = stabilized_weights(treatment, scores)
    np.testing.assert_allclose(weights, [0.625, 5 / 6, 5 / 6, 0.625])
    assert effective_sample_size(np.ones(4)) == 4
    trimmed, low, high = trim_weights(np.array([1.0, 2.0, 3.0, 100.0]))
    assert trimmed.max() == high and trimmed.min() == low
