"""Accumulate collection statistics without retaining individual records."""

from collections import Counter
from dataclasses import dataclass, field, fields
from typing import cast

from ..models import Lemma, Quotation, Sense, Sentence
from .offsets import (
    check_offsets,
    classify_headword,
    compare_sources,
    contains_headword,
)


@dataclass(slots=True)
class Statistics:
    """
    Counts and frequency distributions from a complete collection.

    Distributions retain exact medians while memory follows distinct values.

    Attributes:
        entries: Entry counts by part of speech.
        senses: Sense counts by part of speech.
        senses_per_entry: Distribution of sense counts per entry.
        sentences: Sentence counts by part of speech.
        sentence_kinds: Counts of examples and quotations.
        gloss_depths: Distribution of gloss chain lengths.
        unattested_senses: Senses without sentences.
        quotation_years: Counts of dated quotations by year.
        undated_quotations: Quotations without a year.
        offsets_per_sentence: Distribution of candidate counts per sentence.
        offsets: Total candidate offsets.
        offset_sources: Candidate counts by supporting source combination.
        source_relations: Located sentences by source agreement category.
        unlocated_sentences: Sentences without offsets by part of speech.
        unlocated_headwords: Unlocated sentences by headword shape.
        literal_misses: Unlocated sentences containing the normalized headword.
        offset_violations: Structural violations counted separately by source.
        different_surfaces: Offsets whose text differs from the headword, ignoring case.
        tags: Sense tag occurrences.
        topics: Sense topic occurrences.
        wikidata_ids_per_sense: Distribution of distinct Wikidata IDs per sense.
        variant_entries: Entries with alternative spellings.
        variants: Total alternative spelling links.
        synonym_senses: Senses with synonyms.
        synonyms: Total sense-level synonyms.
        translated_entries: Entries carrying translation tables.
        translation_tables: Total translation tables.
        translations: Words counted per table and language.
        translation_languages: Translation counts by language.
    """

    entries: Counter[str] = field(default_factory=Counter)

    senses: Counter[str] = field(default_factory=Counter)
    senses_per_entry: Counter[int] = field(default_factory=Counter)

    sentences: Counter[str] = field(default_factory=Counter)
    sentence_kinds: Counter[str] = field(default_factory=Counter)

    gloss_depths: Counter[int] = field(default_factory=Counter)

    unattested_senses: int = 0
    quotation_years: Counter[int] = field(default_factory=Counter)
    undated_quotations: int = 0

    offsets_per_sentence: Counter[int] = field(default_factory=Counter)
    offsets: int = 0
    offset_sources: Counter[str] = field(default_factory=Counter)
    source_relations: Counter[str] = field(default_factory=Counter)
    unlocated_sentences: Counter[str] = field(default_factory=Counter)
    unlocated_headwords: Counter[str] = field(default_factory=Counter)
    literal_misses: int = 0
    offset_violations: Counter[str] = field(default_factory=Counter)
    different_surfaces: int = 0

    tags: Counter[str] = field(default_factory=Counter)
    topics: Counter[str] = field(default_factory=Counter)

    wikidata_ids_per_sense: Counter[int] = field(default_factory=Counter)

    variant_entries: int = 0
    variants: int = 0

    synonym_senses: int = 0
    synonyms: int = 0

    translated_entries: int = 0
    translation_tables: int = 0
    translations: int = 0
    translation_languages: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        entry: Lemma,
    ) -> None:
        """
        Count one collected entry and its evidence.

        Args:
            entry: The same entry passed to the JSONL writer.
        """
        self.entries[entry.pos] += 1

        self.senses_per_entry[len(entry.senses)] += 1

        self.variant_entries += bool(entry.variants)
        self.variants += len(entry.variants)

        self.translated_entries += bool(entry.translation_tables)
        self.translation_tables += len(entry.translation_tables)

        for table in entry.translation_tables:
            for language, words in table.translations.items():
                self.translation_languages[language] += len(words)
                self.translations += len(words)

        for sense in entry.senses:
            self.senses[entry.pos] += 1
            self._count_sense(sense)

            for sentence in sense.sentences:
                self._count_sentence(entry, sentence)

    def _count_sense(
        self,
        sense: Sense,
    ) -> None:
        """
        Count sense-level labels, links, and evidence coverage.

        Args:
            sense: Collected sense whose statistics are accumulated.
        """
        self.gloss_depths[sense.depth] += 1

        self.tags.update(sense.tags)
        self.topics.update(sense.topics)

        self.wikidata_ids_per_sense[len(set(sense.wikidata_ids))] += 1

        self.synonym_senses += bool(sense.synonyms)
        self.synonyms += len(sense.synonyms)

        self.unattested_senses += not sense.sentences

    def _count_sentence(
        self,
        entry: Lemma,
        sentence: Sentence,
    ) -> None:
        """
        Count sentence evidence and inspect the proposed word offsets.

        Args:
            entry: Owning entry supplying the headword and part of speech.
            sentence: Example or quotation whose statistics are accumulated.
        """
        self.sentences[entry.pos] += 1

        quoted = isinstance(sentence, Quotation)
        self.sentence_kinds["Quotation" if quoted else "Example"] += 1

        if isinstance(sentence, Quotation):
            if sentence.year is None:
                self.undated_quotations += 1
            else:
                self.quotation_years[sentence.year] += 1

        offsets = sentence.word_offsets
        self.offsets_per_sentence[len(offsets)] += 1

        if not offsets:
            self.unlocated_sentences[entry.pos] += 1
            self.unlocated_headwords[classify_headword(entry.lemma)] += 1

            self.literal_misses += contains_headword(entry.lemma, sentence.text)

            return

        self.source_relations[compare_sources(offsets)] += 1

        self.offset_violations.update(check_offsets(sentence.text, offsets))

        self.offsets += len(offsets)

        self.offset_sources.update(
            "+".join(sorted(offset.sources)) for offset in offsets
        )

        self.different_surfaces += sum(
            sentence.text[start:end].casefold() != entry.lemma.casefold()
            for offset in offsets
            for start, end in (offset.offset,)
        )

    def to_dict(
        self,
    ) -> dict[str, object]:
        """
        Export numeric counts and complete distributions for machine analysis.

        Returns:
            Statistics with JSON-compatible distribution keys and explicit totals.
        """
        report: dict[str, object] = {}

        for attribute in fields(self):
            value = cast(
                Counter[str] | Counter[int] | int, getattr(self, attribute.name)
            )

            report[attribute.name] = (
                {str(key): frequency for key, frequency in sorted(value.items())}
                if isinstance(value, Counter)
                else value
            )

        report["totals"] = {
            "entries": self.entries.total(),
            "senses": self.senses.total(),
            "sentences": self.sentences.total(),
            "offsets": self.offsets,
            "tags": len(self.tags),
            "topics": len(self.topics),
            "translation_languages": len(self.translation_languages),
        }

        return report


def distribution_median(
    frequencies: Counter[int],
) -> float | None:
    """
    Read an exact median from a frequency distribution.

    Args:
        frequencies: Counts of each observed integer.

    Returns:
        The median, or None for an empty distribution.
    """
    total = frequencies.total()

    if not total:
        return None

    lower = (total - 1) // 2
    upper = total // 2

    cumulative = 0
    middle: list[int] = []

    for value, frequency in sorted(frequencies.items()):
        following = cumulative + frequency

        middle.extend(
            value for rank in (lower, upper) if cumulative <= rank < following
        )

        cumulative = following

        if cumulative > upper:
            break

    return sum(middle) / 2
