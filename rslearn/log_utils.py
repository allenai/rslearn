"""Logging utilities."""

import logging
import os
import sys

LOG_FORMAT = "format=%(asctime)s loglevel=%(levelname)-6s logger=%(name)s %(funcName)s() L%(lineno)-4d %(message)s"
# DETAILED_LOG_FORMAT = "format=%(asctime)s loglevel=%(levelname)-6s logger=%(name)s %(funcName)s() L%(lineno)-4d %(message)s   call_trace=%(pathname)s L%(lineno)-4d"  # noqa


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a console handler."""
    this_logger = logging.getLogger(name)
    log_level = os.environ.get("RSLEARN_LOGLEVEL", "INFO")
    if not this_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        this_logger.addHandler(console_handler)

    this_logger.setLevel(log_level)
    this_logger.propagate = True
    return this_logger
