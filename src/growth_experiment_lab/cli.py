"""Command-line interface for Growth Experiment Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from growth_experiment_lab import __version__
from growth_experiment_lab.config import DEFAULT_CONFIG_PATH, ConfigError, load_config
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

    parser.error(f"Unknown command: {args.command}")
    return 2
