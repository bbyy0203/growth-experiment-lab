import json

import matplotlib.pyplot as plt
import numpy as np

from growth_experiment_lab.portfolio import build_portfolio_report, export_examples


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_portfolio_report_complete_and_truth_free(tmp_path):
    quality = tmp_path / "quality.json"
    design = tmp_path / "design.json"
    diagnostics = tmp_path / "diagnostics.json"
    ab = tmp_path / "ab.json"
    obs = tmp_path / "obs.json"
    _write(
        quality,
        {
            "checks_passed": True,
            "datasets": {x: {"total_users": 10} for x in ("randomized", "observational")},
        },
    )
    _write(
        design,
        {
            "primary_metric": "next_day_retention",
            "planned_design": {"total_n": 20, "absolute_effect": 0.01},
            "current_randomized_mart": {"detectable_absolute_effect": 0.02},
        },
    )
    _write(
        diagnostics,
        {
            "srm": {"passed": True},
            "pre_treatment_balance": {"maximum_absolute_smd": 0.01},
            "aa_simulation": {"skipped": True},
        },
    )
    metrics = {
        name: {
            "absolute_effect": 0.01,
            "confidence_interval_lower": 0,
            "confidence_interval_upper": 0.02,
            "raw_p_value": 0.1,
        }
        for name in ("next_day_retention", "clicked", "watch_time_minutes", "negative_feedback")
    }
    metrics["negative_feedback"]["guardrail_status"] = "inconclusive"
    _write(ab, {"metrics": metrics, "cuped": {"variance_reduction_percent": 2}})
    effects = {
        name: {"naive_effect": 0.1, "iptw_ate": 0.01, "trimmed_iptw_ate": 0.02} for name in metrics
    }
    _write(
        obs,
        {
            "effects": effects,
            "overlap": {"overall": {"min": 0.1, "max": 0.9}},
            "weights": {"effective_sample_size": {"overall": 8}},
            "balance": {
                "before": {"maximum_absolute_smd": 0.3},
                "iptw": {"maximum_absolute_smd": 0.02},
            },
        },
    )
    output = tmp_path / "portfolio.md"
    config = {
        "portfolio": {"report_path": str(output), "config_label": "ci"},
        "data_quality": {"report_path": str(quality)},
        "experiment_design": {"report_path": str(design)},
        "randomization_diagnostics": {"report_path": str(diagnostics)},
        "ab_analysis": {"results_path": str(ab)},
        "observational": {"results_path": str(obs)},
    }
    build_portfolio_report(config)
    text = output.read_text(encoding="utf-8").lower()
    assert all(
        section in text
        for section in (
            "analysis flow",
            "experiment design",
            "randomized a/b",
            "observational causal",
            "requires_human_review",
        )
    )
    assert not any(
        token in text
        for token in ("true_sample_ate", "assignment_probability", "potential_outcome")
    )


def test_export_examples_only_approved_artifacts(tmp_path):
    generated = tmp_path / "generated"
    examples = tmp_path / "examples"
    names = {
        "portfolio": "portfolio.md",
        "ab": "ab.md",
        "obs": "obs.md",
        "random": "random.png",
        "effect": "effect.png",
        "overlap": "overlap.png",
        "love": "love.png",
    }
    for name in names.values():
        path = generated / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".md":
            path.write_text("text", encoding="utf-8")
        else:
            plt.imsave(path, np.zeros((2, 2, 3)))
    (generated / "simulation_validation.json").write_text("truth", encoding="utf-8")
    config = {
        "portfolio": {
            "config_label": "demo",
            "examples_dir": str(examples),
            "report_path": str(generated / names["portfolio"]),
        },
        "ab_analysis": {
            "markdown_path": str(generated / names["ab"]),
            "effect_plot_path": str(generated / names["effect"]),
        },
        "observational": {
            "markdown_path": str(generated / names["obs"]),
            "overlap_plot_path": str(generated / names["overlap"]),
            "balance_plot_path": str(generated / names["love"]),
        },
        "randomization_diagnostics": {"balance_plot_path": str(generated / names["random"])},
    }
    outputs = export_examples(config)
    assert len(outputs) == 7 and not (examples / "simulation_validation.json").exists()
    for path in examples.glob("*.md"):
        text = path.read_text(encoding="utf-8").lower()
        assert "generated from synthetic data" in text
        assert "assignment_probability" not in text and "true_sample_ate" not in text
