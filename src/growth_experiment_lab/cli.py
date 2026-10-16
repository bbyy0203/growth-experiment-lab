"""Command-line interface for Growth Experiment Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from growth_experiment_lab import __version__
from growth_experiment_lab.aa_simulation import AASimulationError
from growth_experiment_lab.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from growth_experiment_lab.data_validation import DataValidationError
from growth_experiment_lab.experiment_design import ExperimentDesignError
from growth_experiment_lab.mart import (
    MartBuildError,
    build_all_marts,
    validate_data_and_write_report,
)
from growth_experiment_lab.randomization_diagnostics import RandomizationDiagnosticError
from growth_experiment_lab.stage3 import (
    Stage3OutputError,
    diagnose_randomization,
    plan_experiment,
)
from growth_experiment_lab.synthetic_data import DataGenerationError, generate_and_write


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-1 command-line parser."""
    parser = argparse.ArgumentParser(
        prog="growth-experiment-lab",
        description="Validate configuration for the synthetic Growth Experiment Lab project.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="load and validate the TOML project configuration",
    )
    validate_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )

    generate_parser = subparsers.add_parser(
        "generate-data",
        help="generate randomized and observational synthetic datasets",
    )
    generate_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated files",
    )

    build_parser = subparsers.add_parser(
        "build-mart",
        help="build validated randomized and observational user-level marts",
    )
    build_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing processed mart files",
    )

    validate_data_parser = subparsers.add_parser(
        "validate-data",
        help="validate raw tables and processed marts and write a JSON summary",
    )
    validate_data_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )

    plan_parser = subparsers.add_parser(
        "plan-experiment",
        help="plan the binary primary metric and calculate the current detectable effect",
    )
    plan_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )
    plan_parser.add_argument(
        "--overwrite", action="store_true", help="replace existing JSON output"
    )

    diagnostic_parser = subparsers.add_parser(
        "diagnose-randomization",
        help="run SRM, pre-treatment balance, and independent A/A diagnostics",
    )
    diagnostic_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the TOML configuration (default: %(default)s)",
    )
    diagnostic_parser.add_argument(
        "--overwrite", action="store_true", help="replace existing diagnostic outputs"
    )
    diagnostic_parser.add_argument(
        "--skip-aa", action="store_true", help="skip the independent A/A simulation"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "validate-config":
        try:
            load_config(args.config)
        except ConfigError as error:
            parser.error(str(error))
        print(f"Configuration is valid: {args.config}")
        return 0


    if args.command == "generate-data":
        try:
            config = load_config(args.config)
            result, _ = generate_and_write(config, overwrite=args.overwrite)
        except (ConfigError, DataGenerationError) as error:
            parser.error(str(error))
        print(f"Configuration: {args.config}")
        print(f"Random seed: {config['project']['random_seed']}")
        print(f"Users per dataset: {config['experiment']['n_users']}")
        print(f"Output directory: {config['data_generation']['output_dir']}")
        for dataset_type, dataset in result.datasets.items():
            treatment_share = dataset.assignment["treatment"].mean()
            print(f"Dataset: {dataset_type}; actual treatment share: {treatment_share:.6f}")
        return 0

    if args.command == "build-mart":
        try:
            config = load_config(args.config)
            results = build_all_marts(config, overwrite=args.overwrite)
        except (ConfigError, DataValidationError, MartBuildError) as error:
            parser.error(str(error))
        for result in results:
            print(f"Dataset: {result.dataset_type}")
            for table_name, input_path in result.input_paths.items():
                print(f"Input {table_name}: {input_path}")
            print(f"Output: {result.output_path}")
            print(f"Rows: {len(result.mart)}; columns: {len(result.mart.columns)}")
        return 0

    if args.command == "validate-data":
        try:
            config = load_config(args.config)
            report = validate_data_and_write_report(config)
        except (ConfigError, DataValidationError, MartBuildError) as error:
            parser.error(str(error))
        for dataset_type, summary in report["datasets"].items():
            print(
                f"Dataset: {dataset_type}; users: {summary['total_users']}; "
                f"treatment share: {summary['treatment_share']:.6f}; checks: passed"
            )
        print(f"JSON report: {config['data_quality']['report_path']}")
        return 0

    if args.command == "plan-experiment":
        try:
            config = load_config(args.config)
            report = plan_experiment(config, overwrite=args.overwrite)
        except (
            ConfigError,
            DataValidationError,
            ExperimentDesignError,
            Stage3OutputError,
        ) as error:
            parser.error(str(error))
        planned = report["planned_design"]
        current = report["current_randomized_mart"]
        print(f"Primary metric: {report['primary_metric']}")
        print(
            f"Baseline: {planned['baseline_probability']:.6f}; target absolute MDE: "
            f"{planned['absolute_effect']:.6f}; relative lift: {planned['relative_lift']:.6f}"
        )
        print(
            f"Alpha: {planned['alpha']}; power: {planned['power']}; "
            f"treatment share: {planned['treatment_share']}"
        )
        print(
            f"Required control/treatment/total: {planned['control_n']}/"
            f"{planned['treatment_n']}/{planned['total_n']}"
        )
        print(
            f"Current control/treatment: {current['control_n']}/{current['treatment_n']}; "
            f"detectable absolute MDE: {current['detectable_absolute_effect']:.6f}; "
            f"relative lift: {current['detectable_relative_lift']:.6f}"
        )
        print(f"JSON report: {config['experiment_design']['report_path']}")
        return 0

    if args.command == "diagnose-randomization":
        try:
            config = load_config(args.config)
            report = diagnose_randomization(
                config, overwrite=args.overwrite, skip_aa=args.skip_aa
            )
        except (
            AASimulationError,
            ConfigError,
            DataValidationError,
            RandomizationDiagnosticError,
            Stage3OutputError,
        ) as error:
            parser.error(str(error))
        srm = report["srm"]
        balance = report["pre_treatment_balance"]
        print(
            f"SRM passed: {srm['passed']}; p-value: {srm['p_value']:.6g}; "
            f"observed treatment share: {srm['observed_treatment_share']:.6f}"
        )
        print(
            f"Maximum absolute SMD: {balance['maximum_absolute_smd']:.6f}; "
            f"warning variables: {balance['warning_variables']}"
        )
        if report["aa_simulation"].get("skipped"):
            print("A/A simulation: skipped")
        else:
            aa = report["aa_simulation"]
            print(
                f"A/A false positives: {aa['false_positive_count']}/{aa['simulations']}; "
                f"rate: {aa['empirical_false_positive_rate']:.6f}"
            )
        print(f"Warnings: {report['warnings']}")
        print(f"JSON report: {config['randomization_diagnostics']['report_path']}")
        return 1 if report["critical_failures"] else 0

    parser.error(f"Unknown command: {args.command}")
    return 2
