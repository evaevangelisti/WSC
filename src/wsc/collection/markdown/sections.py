"""
Build the tables for each collection report section.
"""

from ...models import POS
from ..statistics import Statistics, distribution_median
from .format import Table, count_table, format_average

LABEL_LIMIT = 10

_PART_NAMES = {
    POS.NOUN: "Noun",
    POS.PROPN: "Proper noun",
    POS.VERB: "Verb",
    POS.ADJECTIVE: "Adjective",
    POS.ADVERB: "Adverb",
}


def record_tables(
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
        count_table(
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
        count_table(
            "Gloss chain depth",
            (
                (str(depth), count)
                for depth, count in sorted(statistics.gloss_depths.items())
            ),
            senses,
        ),
        count_table(
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


def lexical_link_tables(
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
        count_table(
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
        count_table(
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
                    format_average(statistics.variants, statistics.variant_entries),
                ),
                (
                    "Synonyms per sense with synonyms",
                    format_average(statistics.synonyms, statistics.synonym_senses),
                ),
            ),
        ),
    )


def translation_tables(
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
        count_table(
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
                    format_average(
                        statistics.translations, statistics.translated_entries
                    ),
                ),
            ),
        ),
        count_table(
            "Most frequent translation languages",
            statistics.translation_languages.most_common(LABEL_LIMIT),
            statistics.translations,
        ),
    )


def sentence_tables(
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
        count_table(
            "Sentence kinds",
            statistics.sentence_kinds.most_common(),
            statistics.sentences.total(),
        ),
        count_table(
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


def offset_tables(
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
        count_table(
            "By sentence",
            (("With offsets", sentences - unlocated), ("Without offsets", unlocated)),
            sentences,
        ),
        count_table(
            "Sources by located sentence",
            statistics.source_relations.most_common(),
            sentences - unlocated,
        ),
        count_table(
            "Sources by offset",
            statistics.offset_sources.most_common(),
            statistics.offsets,
        ),
        count_table(
            "Candidate offsets per sentence",
            (
                (str(number), count)
                for number, count in sorted(statistics.offsets_per_sentence.items())
            ),
            sentences,
        ),
        count_table(
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
        count_table(
            "Unlocated headword shapes",
            statistics.unlocated_headwords.most_common(),
            unlocated,
        ),
        count_table(
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
