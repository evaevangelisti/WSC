"""
Format collection statistics as Markdown table cells.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Table:
    """
    A caption, column headings, and formatted rows.

    Attributes:
        caption: Table title within its report section.
        columns: Column headings in display order.
        rows: Formatted cell values in display order.
    """

    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def format_share(
    part: int,
    whole: int,
) -> str:
    """
    Format a percentage when its denominator is nonzero.

    Args:
        part: Count represented by the percentage.
        whole: Total defining the percentage denominator.

    Returns:
        Percentage with one decimal place, or an em dash for a zero total.
    """
    return f"{part / whole:.1%}" if whole else "—"


def format_average(
    total: int,
    observations: int,
) -> str:
    """
    Format an average when there are observations.

    Args:
        total: Sum of the observed values.
        observations: Number of observations contributing to the sum.

    Returns:
        Average with one decimal place, or an em dash for no observations.
    """
    return f"{total / observations:.1f}" if observations else "—"


def count_table(
    caption: str,
    values: Iterable[tuple[str, int]],
    total: int,
) -> Table:
    """
    Build a count table with an explicit percentage denominator.

    Args:
        caption: Table title within its report section.
        values: Row labels and their counts in display order.
        total: Denominator shared by all row percentages.

    Returns:
        A table containing labels, formatted counts, and percentages.
    """
    return Table(
        caption,
        ("Figure", "Count", "Share"),
        tuple(
            (name, f"{value:,}", format_share(value, total)) for name, value in values
        ),
    )


def escape_cell(
    value: str,
) -> str:
    """
    Preserve table boundaries when a label contains Markdown characters.

    Args:
        value: Unescaped cell text from a report row.

    Returns:
        Text with escaped backslashes and pipes, and line breaks replaced by spaces.
    """
    return (
        value.replace("\\", "\\\\")
        .replace("|", r"\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )
