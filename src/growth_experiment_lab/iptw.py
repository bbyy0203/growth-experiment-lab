"""Balance diagnostics and Hajek IPTW effect estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from growth_experiment_lab.propensity import CATEGORICAL_FEATURES, NUMERIC_FEATURES, OUTCOMES


class IPTWError(ValueError):
    """Raised when an IPTW estimate is undefined."""


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(x, weights=w))
    variance = float(np.average(np.square(x - mean), weights=w))
    return mean, variance


def balance_table(frame: pd.DataFrame, weights: np.ndarray | None = None) -> list[dict[str, Any]]:
    t = frame["treatment"].to_numpy() == 1
    w = np.ones(len(frame)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    expanded: list[tuple[str, np.ndarray]] = [
        (x, frame[x].to_numpy(float)) for x in NUMERIC_FEATURES
    ]
    for column in CATEGORICAL_FEATURES:
        expanded.extend(
            (f"{column}={level}", (frame[column].astype(str) == str(level)).to_numpy(float))
            for level in sorted(frame[column].astype(str).unique())
        )
    for name, values in expanded:
        mt, vt = _weighted_mean_var(values[t], w[t])
        mc, vc = _weighted_mean_var(values[~t], w[~t])
        denominator = np.sqrt((vt + vc) / 2)
        smd = 0.0 if denominator == 0 and mt == mc else (mt - mc) / denominator
        rows.append(
            {
                "covariate": name,
                "treatment_mean": mt,
                "control_mean": mc,
                "treatment_variance": vt,
                "control_variance": vc,
                "smd": float(smd),
                "absolute_smd": float(abs(smd)),
            }
        )
    return rows


def summarize_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": rows,
        "maximum_absolute_smd": max(row["absolute_smd"] for row in rows),
        "above_0_10": [row["covariate"] for row in rows if row["absolute_smd"] >= 0.10],
    }


def hajek_effect(frame: pd.DataFrame, outcome: str, weights: np.ndarray) -> dict[str, Any]:
    if outcome not in OUTCOMES:
        raise IPTWError(f"Unsupported outcome: {outcome}")
    t = frame["treatment"].to_numpy() == 1
    if not t.any() or t.all():
        raise IPTWError("Both treatment groups are required")
    y = frame[outcome].to_numpy(float)
    treated = float(np.average(y[t], weights=weights[t]))
    control = float(np.average(y[~t], weights=weights[~t]))
    return {"control_mean": control, "treatment_mean": treated, "ate": treated - control}


def estimate_all_effects(
    frame: pd.DataFrame, weights: np.ndarray, trimmed: np.ndarray
) -> dict[str, Any]:
    unit = np.ones(len(frame))
    results = {}
    for outcome in OUTCOMES:
        naive, adjusted, sensitivity = (
            hajek_effect(frame, outcome, item) for item in (unit, weights, trimmed)
        )
        results[outcome] = {
            "metric_name": outcome,
            "outcome_type": "continuous" if outcome == "watch_time_minutes" else "binary",
            "estimate_scale": "mean_difference"
            if outcome == "watch_time_minutes"
            else "absolute_risk_difference",
            "naive_control_mean": naive["control_mean"],
            "naive_treatment_mean": naive["treatment_mean"],
            "naive_effect": naive["ate"],
            "iptw_control_mean": adjusted["control_mean"],
            "iptw_treatment_mean": adjusted["treatment_mean"],
            "iptw_ate": adjusted["ate"],
            "trimmed_iptw_ate": sensitivity["ate"],
            "naive_minus_iptw": naive["ate"] - adjusted["ate"],
            "iptw_minus_trimmed": adjusted["ate"] - sensitivity["ate"],
            "interpretation": (
                "treatment minus control; adjusted for measured pre-treatment covariates"
            ),
        }
    return results
