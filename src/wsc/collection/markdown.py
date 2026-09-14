"""Render collection statistics as readable Markdown tables."""

from collections.abc import Iterable
from dataclasses import dataclass

from ..models import POS
from .statistics import Statistics, distribution_median

_LABEL_LIMIT = 10

_PART_NAMES = {
    POS.NOUN: "Noun",
    POS.PROPN: "Proper noun",
    POS.VERB: "Verb",
    POS.ADJECTIVE: "Adjective",
    POS.ADVERB: "Adverb",
}


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


def _share(
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


def _average(
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


def _counts(
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
        tuple((name, f"{value:,}", _share(value, total)) for name, value in values),
    )


def _records(
    statistics: Statistics,
) -> tuple[Table, ...]:
    """
    Summarize record counts, gloss depth, and Wikidata coverage.

    Args:
        statistics: Counts and distributions from the complete collection.

    Returns:
        Record totals and coverage tables in display order.
    """
    entries = statistics.entries.total()
    senses = statistics.senses.total()
    sense_median = distribution_median(statistics.senses_per_entry)
    identifiers = statistics.wikidata_ids_per_sense

    return (
        Table(
            "Totals",
            ("Figure", "Count"),
            (
                ("Entries", f"{entries:,}"),
                ("Senses", f"{senses:,}"),
                ("Sentences", f"{statistics.sentences.total():,}"),
                ("Distinct tags", f"{len(statistics.tags):,}"),
                ("Distinct topics", f"{len(statistics.topics):,}"),
            ),
        ),
        Table(
            "Senses per entry",
            ("Statistic", "Value"),
            (
                ("Median", str(sense_median) if entries else "—"),
                (
                    "Maximum",
                    str(max(statistics.senses_per_entry)) if entries else "—",
                ),
            ),
        ),
        _counts(
            "Sentence coverage by sense",
            (
                ("With sentences", senses - statistics.unattested_senses),
                ("Without sentences", statistics.unattested_senses),
            ),
            senses,
        ),
        Table(
            "By part of speech",
            ("Part of speech", "Entries", "Senses", "Sentences", "Unlocated"),
            tuple(
                (
                    name,
                    f"{statistics.entries[part]:,}",
                    f"{statistics.senses[part]:,}",
                    f"{statistics.sentences[part]:,}",
                    f"{statistics.unlocated_sentences[part]:,}",
                )
                for part, name in _PART_NAMES.items()
                if part in statistics.entries
            ),
        ),
        _counts(
            "Gloss chain depth",
            (
                (str(depth), count)
                for depth, count in sorted(statistics.gloss_depths.items())
            ),
            senses,
        ),
        _counts(
            "Wikidata IDs per sense",
            (
                ("None", identifiers[0]),
                ("One", identifiers[1]),
                (
                    "Multiple",
                    sum(count for number, count in identifiers.items() if number > 1),
                ),
            ),
            senses,
        ),
    )


def _lexical_links(
    statistics: Statistics,
) -> tuple[Table, ...]:
    """
    Summarize alternative spellings and sense-level synonyms.

    Args:
        statistics: Counts and distributions from the complete collection.

    Returns:
        Variant and synonym coverage, totals, and averages.
    """
    return (
        _counts(
            "Variants by entry",
            (
                ("With variants", statistics.variant_entries),
                (
                    "Without variants",
                    statistics.entries.total() - statistics.variant_entries,
                ),
            ),
            statistics.entries.total(),
        ),
        _counts(
            "Synonyms by sense",
            (
                ("With synonyms", statistics.synonym_senses),
                (
                    "Without synonyms",
                    statistics.senses.total() - statistics.synonym_senses,
                ),
            ),
            statistics.senses.total(),
        ),
        Table(
            "Totals",
            ("Figure", "Count"),
            (
                ("Variants", f"{statistics.variants:,}"),
                ("Synonyms", f"{statistics.synonyms:,}"),
            ),
        ),
        Table(
            "Averages",
            ("Figure", "Average"),
            (
                (
                    "Variants per entry with variants",
                    _average(statistics.variants, statistics.variant_entries),
                ),
                (
                    "Synonyms per sense with synonyms",
                    _average(statistics.synonyms, statistics.synonym_senses),
                ),
            ),
        ),
    )


def _translations(
    statistics: Statistics,
) -> tuple[Table, ...]:
    """
    Summarize translation tables and their language coverage.

    Args:
        statistics: Counts and distributions from the complete collection.

    Returns:
        Translation coverage, totals, and leading language counts.
    """
    return (
        _counts(
            "By entry",
            (
                ("With translation tables", statistics.translated_entries),
                (
                    "Without translation tables",
                    statistics.entries.total() - statistics.translated_entries,
                ),
            ),
            statistics.entries.total(),
        ),
        Table(
            "Totals",
            ("Figure", "Count"),
            (
                ("Translations", f"{statistics.translations:,}"),
                ("Translation tables", f"{statistics.translation_tables:,}"),
                ("Distinct languages", f"{len(statistics.translation_languages):,}"),
            ),
        ),
        Table(
            "Averages",
            ("Figure", "Average"),
            (
                (
                    "Translations per translated entry",
                    _average(statistics.translations, statistics.translated_entries),
                ),
            ),
        ),
        _counts(
            "Most frequent translation languages",
            statistics.translation_languages.most_common(_LABEL_LIMIT),
            statistics.translations,
        ),
    )


def _sentences(
    statistics: Statistics,
) -> tuple[Table, ...]:
    """
    Summarize sentence kinds and quotation dates.

    Args:
        statistics: Counts and distributions from the complete collection.

    Returns:
        Sentence kind counts, dating coverage, and quotation year summaries.
    """
    years = statistics.quotation_years
    dated = years.total()
    year_median = distribution_median(years)

    return (
        _counts(
            "Sentence kinds",
            statistics.sentence_kinds.most_common(),
            statistics.sentences.total(),
        ),
        _counts(
            "Quotation dates",
            (("Dated", dated), ("Undated", statistics.undated_quotations)),
            dated + statistics.undated_quotations,
        ),
        Table(
            "Quotation years",
            ("Figure", "Year"),
            (
                ("Earliest", str(min(years)) if years else "—"),
                ("Latest", str(max(years)) if years else "—"),
                ("Median", f"{year_median:.0f}" if year_median is not None else "—"),
            ),
        ),
    )


def _offsets(
    statistics: Statistics,
) -> tuple[Table, ...]:
    """
    Summarize offset coverage, source agreement, and structural violations.

    Args:
        statistics: Counts and distributions from the complete collection.

    Returns:
        Offset coverage and diagnostic tables in display order.
    """
    sentences = statistics.sentences.total()
    unlocated = statistics.unlocated_sentences.total()

    return (
        _counts(
            "By sentence",
            (("With offsets", sentences - unlocated), ("Without offsets", unlocated)),
            sentences,
        ),
        _counts(
            "Sources by located sentence",
            statistics.source_relations.most_common(),
            sentences - unlocated,
        ),
        _counts(
            "Sources by offset",
            statistics.offset_sources.most_common(),
            statistics.offsets,
        ),
        _counts(
            "Candidate offsets per sentence",
            (
                (str(number), count)
                for number, count in sorted(statistics.offsets_per_sentence.items())
            ),
            sentences,
        ),
        _counts(
            "Offset surface forms, ignoring case",
            (
                (
                    "Same as headword",
                    statistics.offsets - statistics.different_surfaces,
                ),
                ("Different from headword", statistics.different_surfaces),
            ),
            statistics.offsets,
        ),
        _counts(
            "Unlocated headword shapes",
            statistics.unlocated_headwords.most_common(),
            unlocated,
        ),
        _counts(
            "Literal matches among unlocated sentences",
            (("Headword present after normalization", statistics.literal_misses),),
            unlocated,
        ),
        Table(
            "Offset contract violations by source",
            ("Violation", "Count"),
            tuple(
                (name, f"{count:,}")
                for name, count in statistics.offset_violations.most_common()
            )
            or (("None", "0"),),
        ),
    )


def _escape_cell(
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
        ("Records", _records(statistics)),
        ("Variants and synonyms", _lexical_links(statistics)),
        ("Translations", _translations(statistics)),
        ("Sentences and quotations", _sentences(statistics)),
        ("Word offsets", _offsets(statistics)),
        (
            "Tags and topics",
            (
                _counts(
                    "Most common tags",
                    statistics.tags.most_common(_LABEL_LIMIT),
                    statistics.tags.total(),
                ),
                _counts(
                    "Most common topics",
                    statistics.topics.most_common(_LABEL_LIMIT),
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
                f"| {' | '.join(_escape_cell(cell) for cell in row)} |"
                for row in table.rows
            )

            lines.append("")

    return "\n".join(lines)
