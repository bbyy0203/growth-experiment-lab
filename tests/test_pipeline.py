from pathlib import Path

import pytest

import growth_experiment_lab.pipeline as pipeline
from growth_experiment_lab.config import load_config


def test_demo_and_ci_configs_are_valid():
    assert load_config("configs/demo.toml")["portfolio"]["config_label"] == "demo"
    assert load_config("configs/ci.toml")["experiment"]["n_users"] == 3000


def test_run_all_order_and_skip_options(monkeypatch):
    calls = []
    config = load_config("configs/ci.toml")
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    names = {
        "generate_and_write": "generate-data",
        "build_all_marts": "build-mart",
        "validate_data_and_write_report": "validate-data",
        "plan_experiment": "plan-experiment",
        "diagnose_randomization": "diagnose-randomization",
        "analyze_ab": "analyze-ab",
        "analyze_observational": "analyze-observational",
        "build_portfolio_report": "build-portfolio-report",
    }
    for attribute, name in names.items():
        monkeypatch.setattr(
            pipeline, attribute, lambda *args, _name=name, **kwargs: calls.append(_name)
        )
    monkeypatch.setattr(
        pipeline, "validate_simulation_estimates", lambda *a, **k: calls.append("random-truth")
    )
    monkeypatch.setattr(
        pipeline,
        "validate_observational_simulation",
        lambda *a, **k: calls.append("observational-truth"),
    )
    result = pipeline.run_all(
        "configs/ci.toml",
        overwrite=True,
        skip_aa=True,
        skip_bootstrap=True,
        skip_simulation_validation=True,
    )
    assert calls == [names[x] for x in names]
    assert result["skipped_steps"] == [
        "validate-simulation-estimates",
        "validate-observational-simulation",
    ]


def test_run_all_stops_on_failure(monkeypatch):
    config = load_config("configs/ci.toml")
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "generate_and_write", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "build_all_marts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    monkeypatch.setattr(
        pipeline, "validate_data_and_write_report", lambda *a, **k: pytest.fail("continued")
    )
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.run_all(Path("configs/ci.toml"), overwrite=True)
