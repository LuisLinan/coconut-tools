"""
logger_config.py

This module defines a reusable function to configure and return a color-enhanced logger
using the `colorlog` package. The logger is useful for consistent, readable terminal output
across different scripts in a project.

Usage:
    from logger_config import setup_logger
    logger = setup_logger(__name__)
"""

import logging
from colorlog import ColoredFormatter

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
