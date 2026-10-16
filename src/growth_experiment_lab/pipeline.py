"""Simple fail-fast orchestration for the end-to-end portfolio workflow."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from growth_experiment_lab.ab_analysis import analyze_ab
from growth_experiment_lab.config import load_config
from growth_experiment_lab.mart import build_all_marts, validate_data_and_write_report
from growth_experiment_lab.observational_analysis import analyze_observational
from growth_experiment_lab.observational_validation import validate_observational_simulation
from growth_experiment_lab.portfolio import build_portfolio_report
from growth_experiment_lab.simulation_validation import validate_simulation_estimates
from growth_experiment_lab.stage3 import diagnose_randomization, plan_experiment
from growth_experiment_lab.synthetic_data import generate_and_write

PIPELINE_STEPS = [
    "validate-config",
    "generate-data",
    "build-mart",
    "validate-data",
    "plan-experiment",
    "diagnose-randomization",
    "analyze-ab",
    "validate-simulation-estimates",
    "analyze-observational",
    "validate-observational-simulation",
    "build-portfolio-report",
]


def run_all(
    config_path: str | Path,
    overwrite: bool = False,
    skip_aa: bool = False,
    skip_bootstrap: bool = False,
    skip_simulation_validation: bool = False,
) -> dict[str, Any]:
    """Execute every stage in order and stop immediately on failure."""
    started = time.perf_counter()
    holder: dict[str, dict[str, Any]] = {}
    completed: list[str] = []
    skipped: list[str] = []

    def run(name: str, operation: Callable[[], Any]) -> None:
        print(f"[START] {name}")
        try:
            operation()
        except Exception:
            print(f"[FAIL]  {name}")
            print(
                f"[SUMMARY] completed={len(completed)} skipped={len(skipped)} "
                f"failed={name} elapsed={time.perf_counter() - started:.2f}s "
                f"config={config_path} output=reports/generated"
            )
            raise
        completed.append(name)
        print(f"[DONE]  {name}")

    run("validate-config", lambda: holder.setdefault("config", load_config(config_path)))
    config = holder["config"]
    run("generate-data", lambda: generate_and_write(config, overwrite=overwrite))
    run("build-mart", lambda: build_all_marts(config, overwrite=overwrite))
    run("validate-data", lambda: validate_data_and_write_report(config))
    run("plan-experiment", lambda: plan_experiment(config, overwrite=overwrite))
    run("diagnose-randomization", lambda: diagnose_randomization(config, overwrite, skip_aa))
    run("analyze-ab", lambda: analyze_ab(config, overwrite, skip_bootstrap))
    if skip_simulation_validation:
        skipped.append("validate-simulation-estimates")
        print("[SKIP]  validate-simulation-estimates")
    else:
        run(
            "validate-simulation-estimates",
            lambda: validate_simulation_estimates(config, overwrite),
        )
    run("analyze-observational", lambda: analyze_observational(config, overwrite, skip_bootstrap))
    if skip_simulation_validation:
        skipped.append("validate-observational-simulation")
        print("[SKIP]  validate-observational-simulation")
    else:
        run(
            "validate-observational-simulation",
            lambda: validate_observational_simulation(config, overwrite),
        )
    run("build-portfolio-report", lambda: build_portfolio_report(config, overwrite))
    summary = {
        "success": True,
        "completed_steps": completed,
        "skipped_steps": skipped,
        "failed_step": None,
        "elapsed_seconds": time.perf_counter() - started,
        "config": str(config_path),
        "output_directory": "reports/generated",
    }
    print(
        f"[SUMMARY] completed={len(completed)} skipped={len(skipped)} "
        f"elapsed={summary['elapsed_seconds']:.2f}s config={config_path} output=reports/generated"
    )
    return summary
