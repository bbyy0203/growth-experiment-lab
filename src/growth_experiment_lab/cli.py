"""Command-line interface for Growth Experiment Lab."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from growth_experiment_lab import __version__
from growth_experiment_lab.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


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

    parser.error(f"Unknown command: {args.command}")
    return 2

