"""Minimal smoke tests for the stage-1 package skeleton."""

from growth_experiment_lab.cli import main
from growth_experiment_lab.config import load_config


def test_default_config_loads() -> None:
    config = load_config()

    assert config["project"]["random_seed"] == 20261015
    assert config["metrics"]["next_day_retention"]["role"] == "primary"
    assert config["metrics"]["negative_feedback"]["maximum_allowed_increase"] == 0.005


def test_cli_help(capsys: object) -> None:
    exit_code = main([])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "validate-config" in captured.out


def test_cli_validates_default_config(capsys: object) -> None:
    exit_code = main(["validate-config", "--config", "configs/default.toml"])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Configuration is valid" in captured.out

