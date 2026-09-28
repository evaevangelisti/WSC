"""
Configure command logs without changing the application's root logger.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def configure_logging() -> Generator[None]:
    """
    Send package logs to stderr for the lifetime of a command.

    Yields:
        None while the package log handler is active.
    """
    logger = logging.getLogger(__package__)

    previous_level = logger.level
    previous_propagation = logger.propagate

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()

        logger.setLevel(previous_level)
        logger.propagate = previous_propagation
