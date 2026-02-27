# -*- authors : Vincent Roduit -*-
# -*- date : 2025-10-16 -*-
# -*- Last revision: 2025-11-07 by Vincent Roduit -*-
# -*- python version : 3.13.7. -*-
# -*- Description: Util functions -*-

import logging
import os
import random
from pathlib import Path

import matplotlib as mpl
import optuna
import yaml


def read_yml(cfg_file: Path) -> dict:
    """Read a yaml configuration file.

    Args:
        cfg_file (str): Path to the yaml configuration file.

    Returns:
        dict : Configuration parameters as a dictionary.

    """
    with cfg_file.open("r") as file:
        return yaml.safe_load(file)


def set_seed(seed: int = 42) -> None:
    """Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    """
    random.seed(seed)

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_logging(
    level: int = logging.INFO,
    module_name: str | None = None,
    file_output: Path | None = None,
) -> logging.Logger:
    """Set the logging level for the application.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        module_name (str): Name of the module for which to set the logger.
        file_output (Path, optional): If provided, logs will be written to this file.

    Returns:
        logger: Configured logger instance.

    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=file_output,
        filemode="a" if file_output else None,
    )
    # Set specific logger levels if needed
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    mpl.set_loglevel("WARNING")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return logger
