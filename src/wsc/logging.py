"""Configure command logs without changing the application's root logger."""

import logging
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def configure_logging() -> Generator[None]:
    """Send package logs to stderr for the lifetime of a command."""
    logger = logging.getLogger(__package__)

    previous_level = logger.level

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()

        logger.setLevel(previous_level)
