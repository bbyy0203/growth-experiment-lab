"""CLI and output-contract tests for stage-3 planning and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import growth_experiment_lab.cli as cli_module
import growth_experiment_lab.stage3 as stage3_module
from growth_experiment_lab.config import load_config
from growth_experiment_lab.mart import build_all_marts
from growth_experiment_lab.synthetic_data import generate_all, write_generation_result


@pytest.fixture
def stage3_config(tmp_path: Path) -> dict[str, Any]:
    config = load_config()
    config["experiment"]["n_users"] = 800
    config["data_generation"]["output_dir"] = str(tmp_path / "data")
    config["mart"]["sql_dir"] = str(Path("sql").resolve())
    config["experiment_design"]["report_path"] = str(tmp_path / "reports" / "design.json")
    diagnostics = config["randomization_diagnostics"]
    diagnostics["report_path"] = str(tmp_path / "reports" / "diagnostics.json")
    diagnostics["balance_plot_path"] = str(tmp_path / "reports" / "balance.png")
    diagnostics["aa_plot_path"] = str(tmp_path / "reports" / "aa.png")
    config["aa_simulation"]["n_simulations"] = 20
    config["aa_simulation"]["n_users_per_simulation"] = 500
    write_generation_result(generate_all(config), config["data_generation"]["output_dir"])
    build_all_marts(config)
    return config


def test_plan_cli_json_and_overwrite_contract(
    stage3_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: stage3_config)
    arguments = ["plan-experiment", "--config", "unused.toml"]

    assert cli_module.main(arguments) == 0
    report_path = Path(stage3_config["experiment_design"]["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "synthetic_data_only",
        "primary_metric",
        "planned_design",
        "current_randomized_mart",
        "limitations",
    }
    with pytest.raises(SystemExit) as error:
        cli_module.main(arguments)
    assert error.value.code != 0
    assert cli_module.main([*arguments, "--overwrite"]) == 0


def test_diagnostic_cli_skip_aa_and_full_output(
    stage3_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "load_config", lambda _: stage3_config)
    arguments = ["diagnose-randomization", "--config", "unused.toml", "--skip-aa"]

    assert cli_module.main(arguments) == 0
    report_path = Path(stage3_config["randomization_diagnostics"]["report_path"])
    skipped = json.loads(report_path.read_text(encoding="utf-8"))
    assert skipped["aa_simulation"] == {"skipped": True}
    with pytest.raises(SystemExit):
        cli_module.main(arguments)

    full_arguments = [
        "diagnose-randomization",
        "--config",
        "unused.toml",
        "--overwrite",
    ]
    assert cli_module.main(full_arguments) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "synthetic_data_only",
        "dataset_type",
        "critical_failures",
        "warnings",
        "passed_checks",
        "srm",
        "pre_treatment_balance",
        "aa_simulation",
        "generated_plots",
        "scope_note",
    }
    assert report["dataset_type"] == "randomized"
    assert "p_values" not in report["aa_simulation"]


def _stub_diagnostic_report(critical: bool, warning: bool) -> dict[str, Any]:
    return {
        "critical_failures": ["srm_failed"] if critical else [],
        "warnings": ["balance_warning"] if warning else [],
        "srm": {"passed": not critical, "p_value": 0.5, "observed_treatment_share": 0.5},
        "pre_treatment_balance": {
            "maximum_absolute_smd": 0.2 if warning else 0.01,
            "warning_variables": ["x"] if warning else [],
        },
        "aa_simulation": {"skipped": True},
    }


def test_cli_critical_failure_is_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"randomization_diagnostics": {"report_path": "unused.json"}}
    monkeypatch.setattr(cli_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli_module,
        "diagnose_randomization",
        lambda *args, **kwargs: _stub_diagnostic_report(True, False),
    )
    assert cli_module.main(["diagnose-randomization"]) == 1


def test_cli_warning_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"randomization_diagnostics": {"report_path": "unused.json"}}
    monkeypatch.setattr(cli_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli_module,
        "diagnose_randomization",
        lambda *args, **kwargs: _stub_diagnostic_report(False, True),
    )
    assert cli_module.main(["diagnose-randomization"]) == 0


def test_stage3_default_path_does_not_reference_truth_sources() -> None:
    source = Path(stage3_module.__file__).read_text(encoding="utf-8")

    assert "true_effects.json" not in source
    assert "assignment_probability" not in source
    assert 'mart_path(data_root, "randomized")' in source
