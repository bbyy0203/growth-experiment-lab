"""Independent null A/A simulations for false-positive calibration."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "growth_experiment_lab_matplotlib")
)

import matplotlib
import numpy as np
from scipy.stats import binom
from statsmodels.stats.proportion import proportions_ztest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class AASimulationError(ValueError):
    """Raised for invalid A/A simulation parameters."""


def generate_null_sample(
    rng: np.random.Generator,
    users: int,
    baseline_probability: float,
    treatment_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pre-treatment heterogeneity, independent assignment, and a null outcome."""
    activity = rng.normal(0.0, 1.0, users)
    retention_probability = np.clip(baseline_probability + 0.06 * activity, 0.01, 0.99)
    treatment = rng.binomial(1, treatment_share, users)
    retention = rng.binomial(1, retention_probability)
    return activity, treatment, retention, retention_probability


def run_aa_simulation(
    simulations: int,
    users_per_simulation: int,
    alpha: float,
    random_seed: int,
    baseline_probability: float,
    treatment_share: float,
) -> dict[str, Any]:
    """Run repeated two-proportion tests under an explicitly null treatment effect."""
    if not isinstance(simulations, int) or simulations <= 0:
        raise AASimulationError("simulations must be a positive integer")
    if not isinstance(users_per_simulation, int) or users_per_simulation <= 1:
        raise AASimulationError("users_per_simulation must be an integer greater than 1")
    if not 0 < alpha < 1:
        raise AASimulationError("alpha must be between 0 and 1")
    if not 0 < baseline_probability < 1:
        raise AASimulationError("baseline_probability must be between 0 and 1")
    if not 0 < treatment_share < 1:
        raise AASimulationError("treatment_share must be between 0 and 1")

    root = np.random.default_rng(random_seed)
    p_values = np.empty(simulations, dtype=float)
    for index, child in enumerate(root.spawn(simulations)):
        rng = np.random.default_rng(child)
        _, treatment, retention, _ = generate_null_sample(
            rng, users_per_simulation, baseline_probability, treatment_share
        )
        treatment_mask = treatment == 1
        treatment_n = int(treatment_mask.sum())
        control_n = users_per_simulation - treatment_n
        if treatment_n == 0 or control_n == 0:
            raise AASimulationError("an A/A iteration produced an empty assignment group")
        successes = [int(retention[treatment_mask].sum()), int(retention[~treatment_mask].sum())]
        _, p_value = proportions_ztest(successes, [treatment_n, control_n], alternative="two-sided")
        p_values[index] = p_value

    false_positive_count = int((p_values < alpha).sum())
    empirical_rate = false_positive_count / simulations
    lower_count = int(binom.ppf(0.025, simulations, alpha))
    upper_count = int(binom.ppf(0.975, simulations, alpha))
    interval = [lower_count / simulations, upper_count / simulations]
    quantile_levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quantiles = np.quantile(p_values, quantile_levels)
    return {
        "simulations": simulations,
        "users_per_simulation": users_per_simulation,
        "alpha": alpha,
        "false_positive_count": false_positive_count,
        "empirical_false_positive_rate": empirical_rate,
        "expected_false_positive_rate": alpha,
        "monte_carlo_standard_error": math.sqrt(alpha * (1.0 - alpha) / simulations),
        "binomial_95_percent_interval": interval,
        "within_expected_interval": bool(interval[0] <= empirical_rate <= interval[1]),
        "p_value_mean": float(p_values.mean()),
        "p_value_median": float(np.median(p_values)),
        "p_value_quantiles": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles, strict=True)
        },
        "minimum_p_value": float(p_values.min()),
        "maximum_p_value": float(p_values.max()),
        "random_seed": random_seed,
        "p_values": p_values.tolist(),
    }


def save_pvalue_histogram(result: dict[str, Any], output_path: str | Path) -> Path:
    """Save a histogram of null p-values from the synthetic A/A simulation."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(result["p_values"], bins=20, range=(0, 1), color="#72B7B2", edgecolor="white")
    axis.axvline(result["alpha"], color="#E45756", linestyle="--", label="alpha")
    axis.set_xlabel("Two-proportion test p-value")
    axis.set_ylabel("Simulation count")
    axis.set_title("Synthetic null A/A p-value distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
