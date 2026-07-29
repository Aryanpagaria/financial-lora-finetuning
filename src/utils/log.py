"""
Logging utilities for the project.
"""

import logging
from pathlib import Path

from src.utils.config_loader import load_configs


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Create and return a configured logger.
    """

    config = load_configs()

    log_directory = Path("artifacts/logs")

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = log_directory / "training.log"

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(
        formatter
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger