import json

import pytest

from growth_experiment_lab.observational_validation import (
    ObservationalValidationError,
    validate_observational_simulation,
)


def test_validation_requires_formal_results(tmp_path):
    config = {
        "observational": {
            "results_path": str(tmp_path / "missing.json"),
            "simulation_validation_path": str(tmp_path / "out.json"),
        },
        "data_generation": {"output_dir": str(tmp_path)},
    }
    with pytest.raises(ObservationalValidationError, match="do not exist"):
        validate_observational_simulation(config)


def test_validation_is_isolated_and_compares_errors(tmp_path):
    formal = tmp_path / "formal.json"
    truth_dir = tmp_path / "metadata"
    truth_dir.mkdir()
    effects = {
        name: {
            "naive_effect": 0.1,
            "iptw_ate": 0.02,
            "trimmed_iptw_ate": 0.03,
            "bootstrap": {"confidence_interval_lower": 0, "confidence_interval_upper": 0.04},
        }
        for name in ("clicked", "next_day_retention", "negative_feedback", "watch_time_minutes")
    }
    formal.write_text(json.dumps({"effects": effects}), encoding="utf-8")
    (truth_dir / "true_effects.json").write_text(
        json.dumps({"true_sample_ate": {"observational": {name: 0.01 for name in effects}}}),
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"
    config = {
        "observational": {"results_path": str(formal), "simulation_validation_path": str(output)},
        "data_generation": {"output_dir": str(tmp_path)},
    }
    report = validate_observational_simulation(config)
    assert report["synthetic_validation_only"] is True
    assert report["summary"]["iptw_closer_than_naive_count"] == 4
    with pytest.raises(ObservationalValidationError, match="already exists"):
        validate_observational_simulation(config)


def test_validation_allows_skipped_bootstrap(tmp_path):
    formal = tmp_path / "formal.json"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    names = ("clicked", "next_day_retention", "negative_feedback", "watch_time_minutes")
    effects = {
        name: {
            "naive_effect": 0.1,
            "iptw_ate": 0.02,
            "trimmed_iptw_ate": 0.03,
            "bootstrap": None,
        }
        for name in names
    }
    formal.write_text(json.dumps({"effects": effects}), encoding="utf-8")
    (metadata / "true_effects.json").write_text(
        json.dumps({"true_sample_ate": {"observational": {name: 0.01 for name in names}}}),
        encoding="utf-8",
    )
    config = {
        "observational": {
            "results_path": str(formal),
            "simulation_validation_path": str(tmp_path / "validation.json"),
        },
        "data_generation": {"output_dir": str(tmp_path)},
    }
    report = validate_observational_simulation(config)
    assert report["comparisons"]["clicked"]["true_ate_in_iptw_95_percent_ci"] is None
