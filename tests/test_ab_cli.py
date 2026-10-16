"""End-to-end small-fixture tests for formal A/B and isolated truth validation CLIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import growth_experiment_lab.ab_analysis as ab_module
import growth_experiment_lab.cli as cli_module
from growth_experiment_lab.ab_analysis import ABAnalysisError
from growth_experiment_lab.config import load_config
from growth_experiment_lab.mart import build_all_marts
from growth_experiment_lab.synthetic_data import generate_all, write_generation_result


@pytest.fixture
def ab_config(tmp_path: Path) -> dict[str, Any]:
    config = load_config()
    config["experiment"]["n_users"] = 1_200
    config["data_generation"]["output_dir"] = str(tmp_path / "data")
    config["mart"]["sql_dir"] = str(Path("sql").resolve())
    settings = config["ab_analysis"]
    settings["bootstrap_resamples"] = 30
    settings["results_path"] = str(tmp_path / "reports" / "ab_results.json")
    settings["markdown_path"] = str(tmp_path / "reports" / "ab_analysis.md")
    settings["effect_plot_path"] = str(tmp_path / "reports" / "effects.png")
    settings["cuped_plot_path"] = str(tmp_path / "reports" / "cuped.png")
    settings["subgroup_plot_path"] = str(tmp_path / "reports" / "subgroups.png")
    settings["simulation_validation_path"] = str(tmp_path / "reports" / "validation.json")
    write_generation_result(generate_all(config), settings_output_root(config))
    build_all_marts(config)
    return config


def settings_output_root(config: dict[str, Any]) -> str:
    return config["data_generation"]["output_dir"]


def test_analyze_cli_skip_options_overwrite_and_full_fields(
    ab_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: ab_config)
    base = ["analyze-ab", "--config", "unused.toml"]
    skipped_args = [*base, "--skip-bootstrap", "--skip-cuped", "--skip-segments"]

    assert cli_module.main(skipped_args) == 0
    skipped = json.loads(
        Path(ab_config["ab_analysis"]["results_path"]).read_text(encoding="utf-8")
    )
    assert skipped["cuped"] == {"skipped": True}
    assert skipped["predefined_subgroups"] == {"skipped": True}
    assert all(result["bootstrap"] == {"skipped": True} for result in skipped["metrics"].values())
    with pytest.raises(SystemExit):
        cli_module.main(skipped_args)

    assert cli_module.main([*base, "--overwrite"]) == 0
    report = json.loads(
        Path(ab_config["ab_analysis"]["results_path"]).read_text(encoding="utf-8")
    )
    assert set(report) == {
        "synthetic_data_only",
        "not_a_production_experiment",
        "analysis_principle",
        "dataset_type",
        "randomization_quality",
        "metric_roles",
        "metrics",
        "cuped",
        "predefined_subgroups",
        "decision_support",
        "generated_plots",
        "requires_human_review",
    }
    assert set(report["metrics"]) == {
        "next_day_retention",
        "clicked",
        "watch_time_minutes",
        "negative_feedback",
    }
    assert report["requires_human_review"] is True
    assert report["metrics"]["clicked"]["adjusted_p_value"] is not None
    assert report["metrics"]["watch_time_minutes"]["adjusted_p_value"] is not None
    assert report["metrics"]["next_day_retention"]["adjusted_p_value"] is None
    markdown = Path(ab_config["ab_analysis"]["markdown_path"]).read_text(encoding="utf-8")
    assert "synthetic" in markdown.lower()
    assert "human review" in markdown.lower()


def test_formal_outputs_exclude_truth_and_forbidden_assignment_data(
    ab_config: dict[str, Any]
) -> None:
    ab_module.analyze_ab(ab_config, overwrite=True, skip_bootstrap=True)
    json_text = Path(ab_config["ab_analysis"]["results_path"]).read_text(encoding="utf-8").lower()
    markdown = Path(ab_config["ab_analysis"]["markdown_path"]).read_text(encoding="utf-8").lower()
    forbidden = (
        "true_sample_ate",
        "true_ate",
        "true_effect",
        "assignment_probability",
        "potential_outcome",
    )
    assert all(token not in json_text for token in forbidden)
    assert all(token not in markdown for token in forbidden)
    source = Path(ab_module.__file__).read_text(encoding="utf-8")
    assert "true_effects.json" not in source
    assert 'mart_path(Path(config["data_generation"]["output_dir"]), "randomized")' in source


def test_isolated_simulation_validation_requires_results_and_writes_marker(
    ab_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: ab_config)
    results_path = Path(ab_config["ab_analysis"]["results_path"])
    if results_path.exists():
        results_path.unlink()
    with pytest.raises(SystemExit):
        cli_module.main(["validate-simulation-estimates", "--config", "unused.toml"])

    ab_module.analyze_ab(ab_config, overwrite=True, skip_bootstrap=True)
    assert cli_module.main(["validate-simulation-estimates", "--config", "unused.toml"]) == 0
    output = Path(ab_config["ab_analysis"]["simulation_validation_path"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["synthetic_validation_only"] is True
    assert set(report["comparisons"]) == {
        "next_day_retention",
        "clicked",
        "watch_time_minutes",
        "negative_feedback",
    }
    with pytest.raises(SystemExit):
        cli_module.main(["validate-simulation-estimates", "--config", "unused.toml"])
    assert (
        cli_module.main(
            ["validate-simulation-estimates", "--config", "unused.toml", "--overwrite"]
        )
        == 0
    )


@pytest.mark.parametrize("message", ["data validation failed", "SRM failed"])
def test_analyze_cli_structural_failure_is_nonzero(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: {})

    def fail(*args: object, **kwargs: object) -> dict[str, Any]:
        raise ABAnalysisError(message)

    monkeypatch.setattr(cli_module, "analyze_ab", fail)
    with pytest.raises(SystemExit) as error:
        cli_module.main(["analyze-ab"])
    assert error.value.code != 0
