"""Tests for independent null A/A simulation behavior."""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from growth_experiment_lab.aa_simulation import generate_null_sample, run_aa_simulation


def _run(seed: int) -> dict[str, object]:
    return run_aa_simulation(40, 1_000, 0.05, seed, 0.30, 0.50)


def test_same_seed_is_reproducible_and_different_seed_changes_result() -> None:
    first = _run(1234)
    second = _run(1234)
    changed = _run(1235)

    assert first == second
    assert first["p_values"] != changed["p_values"]


def test_all_pvalues_are_valid() -> None:
    result = _run(100)
    p_values = np.asarray(result["p_values"])

    assert np.all((0 <= p_values) & (p_values <= 1))


def test_null_probability_does_not_use_treatment() -> None:
    activity, treatment, _, probability = generate_null_sample(
        np.random.default_rng(7), 10_000, 0.30, 0.50
    )

    assert np.allclose(probability, np.clip(0.30 + 0.06 * activity, 0.01, 0.99))
    assert abs(np.corrcoef(activity, treatment)[0, 1]) < 0.03


def test_false_positive_rate_uses_wide_predefined_acceptance_interval() -> None:
    result = run_aa_simulation(100, 1_000, 0.05, 20261016, 0.30, 0.50)
    lower = binom.ppf(0.005, 100, 0.05) / 100
    upper = binom.ppf(0.995, 100, 0.05) / 100

    assert lower <= result["empirical_false_positive_rate"] <= upper
