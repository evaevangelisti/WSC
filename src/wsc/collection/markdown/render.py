"""
Render collection statistics as readable Markdown tables.
"""

from ..statistics import Statistics
from .format import count_table, escape_cell
from .sections import (
    LABEL_LIMIT,
    lexical_link_tables,
    offset_tables,
    record_tables,
    sentence_tables,
    translation_tables,
)


def render_markdown(
    statistics: Statistics,
) -> str:
    """
    Render all report sections from the collected statistics.

    Args:
        statistics: Complete counts, including empty distributions.

    Returns:
        A Markdown document describing coverage rather than annotation accuracy.
    """
    sections = (
        ("Records", record_tables(statistics)),
        ("Variants and synonyms", lexical_link_tables(statistics)),
        ("Translations", translation_tables(statistics)),
        ("Sentences and quotations", sentence_tables(statistics)),
        ("Word offsets", offset_tables(statistics)),
        (
            "Tags and topics",
            (
                count_table(
                    "Most common tags",
                    statistics.tags.most_common(LABEL_LIMIT),
                    statistics.tags.total(),
                ),
                count_table(
                    "Most common topics",
                    statistics.topics.most_common(LABEL_LIMIT),
                    statistics.topics.total(),
                ),
            ),
        ),
    )

    lines = ["# Collection report", ""]

    for title, tables in sections:
        lines.extend((f"## {title}", ""))

        for table in tables:
            lines.extend((f"### {table.caption}", ""))

            lines.append(f"| {' | '.join(table.columns)} |")
            lines.append(f"| {' | '.join('---' for _ in table.columns)} |")

            lines.extend(
                f"| {' | '.join(escape_cell(cell) for cell in row)} |"
                for row in table.rows
            )

            lines.append("")

    return "\n".join(lines)
