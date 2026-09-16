"""Report long-running progress without flooding a redirected log."""

from __future__ import annotations

import sys

from tqdm import tqdm

TERMINAL_INTERVAL = 0.5
"""Refresh an attached terminal twice a second."""

LOG_INTERVAL = 30.0
"""Refresh a redirected log at most twice a minute."""


def refresh_interval() -> float:
    """
    Choose how often a bar may redraw itself.

    A redirected bar writes one line per refresh, so a batch run keeps its log
    readable while an interactive run stays responsive.

    Returns:
        Minimum seconds between refreshes.
    """
    return TERMINAL_INTERVAL if sys.stderr.isatty() else LOG_INTERVAL


def nested_position() -> int:
    """
    Choose the line a secondary bar occupies.

    Stacked bars steer the cursor with escape sequences, which only a terminal
    interprets, so a redirected bar stays on the first line.

    Returns:
        The line reserved for a bar running inside another one.
    """
    return 1 if sys.stderr.isatty() else 0


def progress_bar(
    description: str,
    unit: str,
    total: int | None = None,
    position: int = 0,
    *,
    leave: bool = True,
) -> tqdm[None]:
    """
    Open a progress bar whose refresh rate suits its output stream.

    Args:
        description: Label shown before the bar.
        unit: Name of one counted item.
        total: Expected number of items, or None when it is unknown.
        position: Line the bar occupies among concurrent bars.
        leave: Whether the finished bar stays on screen.

    Returns:
        A configured progress bar.
    """
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        position=position,
        leave=leave,
        mininterval=refresh_interval(),
        dynamic_ncols=True,
    )
