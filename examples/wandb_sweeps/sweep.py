#!/usr/bin/env python
"""Runs a W&B sweep."""

import argparse
import functools
import logging
import subprocess
import sys
import tempfile
import traceback
import warnings

from typing import Any, Dict, List, TextIO

import yaml
import wandb

warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")


def train_sweep(
    config: Dict[str, Any], temp_config: TextIO, argv: List[str]
) -> None:
    """Runs a single training run.

    Args:
        config: path to UDTube YAML config file.
        temp_config: temporary configuration file handle.
        argv: command-line arguments.
    """
    try:
        populate_config(config, temp_config)
        exit_code = run_sweep(argv)
        wandb.finish(exit_code=exit_code)
    except Exception as e:
        logging.error("Error in train_sweep: %s", str(e))
        logging.error("Full traceback: %s", traceback.format_exc())
        wandb.finish(exit_code=1)


def run_sweep(argv: List[str]) -> int:
    """Actually runs the sweep.

    Args:
        argv: command-line arguments.

    Returns:
        int: The exit code from the subprocess. 0 for success, non-zero for failure.

    We encapsulate each run by using a separate subprocess, which ought to
    ensure that memory is returned (etc.).
    """
    try:
        result = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True
        )
        return 0
    except subprocess.CalledProcessError as error:
        logging.error("Subprocess error: %s", error)
        if error.stdout:
            logging.error("Process stdout:\n%s", error.stdout)
        if error.stderr:
            logging.error("Process stderr:\n%s", error.stderr)
        return error.returncode


def populate_config(
    config: Dict[str, Any], temp_config_handle: TextIO
) -> None:
    """Populates temporary configuration file.

    The wandb config data used here comes from the environment.

    Args:
        config: path to UDTube YAML config file.
        temp_config_handle: temporary configuration file handle.
    """
    wandb.init()
    try:
        for key, value in wandb.config.items():
            _recursive_insert(config, key, value)
        yaml.dump(config, temp_config_handle)
    except Exception as e:
        logging.error("Error populating config: %s", str(e))
        logging.error("Full traceback: %s", traceback.format_exc())
        wandb.finish(exit_code=1)


def _recursive_insert(config: Dict[str, Any], key: str, value: Any) -> None:
    """Recursively inserts values into a nested dictionary.

    Args:
        config: the config dictionary.
        key: a string with the arguments separated by ".".
        value: the value to insert.
    """
    *most, last = key.split(".")
    ptr = config
    for piece in most:
        try:
            ptr = ptr[piece]
        except KeyError:
            ptr[piece] = {}
            ptr = ptr[piece]
    if last in ptr:
        logging.debug(
            "Overriding configuration argument %s with W&B sweep value: %r",
            key,
            value,
        )
    ptr[last] = value


def main(args: argparse.Namespace) -> None:
    try:
        with open(args.config, "r") as source:
            config = yaml.safe_load(source)
        
        # TODO: Consider enabling the W&B logger; we are not sure if things will
        # work unless this is configured.
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml")
        argv = ["udtube", "fit", "--config", temp_config.name, *sys.argv[1:]]
        
        wandb.agent(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            function=functools.partial(train_sweep, config, temp_config, argv),
            count=args.count,
        )
    except Exception as e:
        logging.error("Error in main: %s", str(e))
        logging.error("Full traceback: %s", traceback.format_exc())
        wandb.finish(exit_code=1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(filename)s %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="ID for the sweep.")
    parser.add_argument(
        "--entity", required=True, help="The entity scope for the project."
    )
    parser.add_argument(
        "--project", required=True, help="The project of the sweep."
    )
    parser.add_argument("--count", type=int, help="Number of runs to perform.")
    parser.add_argument("--config", required=True)
    # We pass the known args to main but remove them from ARGV.
    # See: https://docs.python.org/3/library/argparse.html#partial-parsing
    # This allows the user to override config arguments with CLI arguments.
    args, sys.argv[1:] = parser.parse_known_args()
    main(args)
