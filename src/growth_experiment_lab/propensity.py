"""Propensity scores and stabilized IPTW using a fixed pre-treatment contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "is_new_user",
    "pre_7d_sessions",
    "pre_7d_watch_time_minutes",
    "pre_7d_click_days",
    "pre_experiment_activity_score",
]
CATEGORICAL_FEATURES = ["device_type", "region"]
PROPENSITY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OUTCOMES = ["next_day_retention", "clicked", "watch_time_minutes", "negative_feedback"]
FORBIDDEN_FEATURES = set(OUTCOMES) | {
    "assignment_probability",
    "assignment_date",
    "dataset_type",
    "treatment",
    "p0",
    "p1",
    "true_effect",
    "true_ate",
    "potential_outcome",
}


class PropensityError(ValueError):
    """Raised when the propensity or weighting contract is violated."""


def validate_feature_contract(frame: pd.DataFrame, features: list[str] | None = None) -> None:
    selected = PROPENSITY_FEATURES if features is None else features
    forbidden = set(selected) & FORBIDDEN_FEATURES
    if forbidden:
        raise PropensityError(f"Forbidden propensity features: {', '.join(sorted(forbidden))}")
    if set(selected) != set(PROPENSITY_FEATURES):
        raise PropensityError("Propensity features must exactly match the pre-treatment whitelist")
    missing = set(selected + ["treatment"]) - set(frame.columns)
    if missing:
        raise PropensityError(f"Missing propensity columns: {', '.join(sorted(missing))}")
    if frame[selected + ["treatment"]].isna().any().any():
        raise PropensityError("Propensity inputs must not contain missing values")
    if frame["treatment"].nunique() != 2:
        raise PropensityError("Propensity fitting requires both treatment groups")


def fit_propensity(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one unpenalized logistic model and return protected probabilities."""
    validate_feature_contract(frame)
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    pipeline = Pipeline([("transform", transform), ("model", model)])
    try:
        pipeline.fit(frame[PROPENSITY_FEATURES], frame["treatment"])
    except Exception as error:
        raise PropensityError(f"Propensity model failed to converge or fit: {error}") from error
    if int(model.n_iter_[0]) >= model.max_iter:
        raise PropensityError("Propensity model did not converge within max_iter")
    scores = pipeline.predict_proba(frame[PROPENSITY_FEATURES])[:, 1]
    if not np.isfinite(scores).all() or np.any((scores <= 0) | (scores >= 1)):
        raise PropensityError("Propensity scores must be finite and strictly between 0 and 1")
    names = pipeline.named_steps["transform"].get_feature_names_out().tolist()
    return scores, {
        "method": "unpenalized logistic regression (scikit-learn lbfgs)",
        "features": PROPENSITY_FEATURES,
        "design_matrix_columns": names,
        "converged": True,
        "iterations": int(model.n_iter_[0]),
        "sample_size": len(frame),
        "treatment_share": float(frame["treatment"].mean()),
    }


def stabilized_weights(treatment: pd.Series | np.ndarray, scores: np.ndarray) -> np.ndarray:
    treatment_array = np.asarray(treatment, dtype=int)
    if np.any((scores <= 0) | (scores >= 1)):
        raise PropensityError("Cannot weight propensity scores outside (0, 1)")
    prevalence = float(treatment_array.mean())
    weights = np.where(treatment_array == 1, prevalence / scores, (1 - prevalence) / (1 - scores))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise PropensityError("Stabilized weights must be finite and positive")
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def trim_weights(
    weights: np.ndarray, lower: float = 0.01, upper: float = 0.99
) -> tuple[np.ndarray, float, float]:
    low, high = np.quantile(weights, [lower, upper])
    return np.clip(weights, low, high), float(low), float(high)


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    quantiles = {
        str(q): float(np.quantile(values, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    }
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "quantiles": quantiles,
    }


def propensity_diagnostics(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float = 0.05
) -> dict[str, Any]:
    treated = scores[frame["treatment"].to_numpy() == 1]
    control = scores[frame["treatment"].to_numpy() == 0]
    lower, upper = max(treated.min(), control.min()), min(treated.max(), control.max())
    outside = (scores < lower) | (scores > upper)
    treatment_mask = frame["treatment"].to_numpy() == 1
    extreme = (scores < threshold) | (scores > 1 - threshold)
    warnings = []
    if float(extreme.mean()) > 0.01:
        warnings.append("More than 1% of users have propensity below 0.05 or above 0.95")
    if lower >= upper or float(outside.mean()) > 0.10:
        warnings.append("Common support is materially limited")
    return {
        "overall": distribution_summary(scores),
        "treatment": distribution_summary(treated),
        "control": distribution_summary(control),
        "common_support": {
            "lower": float(lower),
            "upper": float(upper),
            "outside_count": int(outside.sum()),
            "outside_share": float(outside.mean()),
            "treatment_outside_count": int(outside[treatment_mask].sum()),
            "treatment_outside_share": float(outside[treatment_mask].mean()),
            "control_outside_count": int(outside[~treatment_mask].sum()),
            "control_outside_share": float(outside[~treatment_mask].mean()),
        },
        "below_0_05_share": float((scores < threshold).mean()),
        "above_0_95_share": float((scores > 1 - threshold).mean()),
        "warnings": warnings,
        "severe_violation": bool(lower >= upper or outside.mean() > 0.25),
    }


def weight_diagnostics(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    t = frame["treatment"].to_numpy() == 1
    result = distribution_summary(weights)
    result.update(
        {
            "coefficient_of_variation": float(weights.std(ddof=1) / weights.mean()),
            "sum": float(weights.sum()),
            "all_finite_positive": bool(np.isfinite(weights).all() and np.all(weights > 0)),
            "treatment": distribution_summary(weights[t]),
            "control": distribution_summary(weights[~t]),
            "extreme_counts": {
                str(x): {"count": int((weights > x).sum()), "share": float((weights > x).mean())}
                for x in (5, 10, 20)
            },
            "effective_sample_size": {
                "overall": effective_sample_size(weights),
                "treatment": effective_sample_size(weights[t]),
                "control": effective_sample_size(weights[~t]),
            },
        }
    )
    return result
