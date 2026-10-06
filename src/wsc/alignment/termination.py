"""
Unwind alignment writers when the scheduler requests termination.
"""

import signal
from collections.abc import Generator
from contextlib import contextmanager
from types import FrameType
from typing import Never


@contextmanager
def handle_termination() -> Generator[None]:
    """
    Convert SIGTERM into an exit that publishes completed cache decisions.

    Yields:
        None while the termination handler is active.
    """

    def terminate(
        signal_number: int,
        frame: FrameType | None,
    ) -> Never:
        """
        Exit through the active contexts so their writers can close.

        Args:
            signal_number: Termination signal received by the process.
            frame: Interrupted execution frame.

        Raises:
            SystemExit: With the conventional signal exit status.
        """
        del frame

        raise SystemExit(128 + signal_number)

    previous_handler = signal.signal(signal.SIGTERM, terminate)

    try:
        yield
    finally:
        _ = signal.signal(signal.SIGTERM, previous_handler)
