"""
Extraction of lemmas from wiktextract output.
"""

import gzip
import json
import re
from collections.abc import Iterator
from compression import zstd
from pathlib import Path
from typing import IO, TypedDict, cast

from tqdm import tqdm

from ...models import (
    POS,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
)
from ..offsets import find_word_offsets

# A year of its own, from the first century of printing up to this one, or
# the decade it opens.
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")

# Rows describing an inflection table rather than the lemma, and the
# transliterations that stand beside a form rather than for it.
_SERVICE_TAGS = frozenset({"inflection-template", "romanization", "table-tags"})

# What an inflection table writes for a cell it leaves empty.
_EMPTY_CELL = "-"


# The four classes below name the slice of the wiktextract schema this
# module reads. Every key is optional, since it describes someone else's JSON.


class _RawExample(TypedDict, total=False):
    """
    One sentence illustrating a sense.

    Attributes:
        text: The sentence.
        ref: The source, when the sentence is quoted from one.
    """

    text: str
    ref: str


class _RawSense(TypedDict, total=False):
    """
    One sense of an entry.

    Attributes:
        glosses: The gloss chain, outermost first.
        tags: Labels of grammar and register.
        topics: Subject fields the sense belongs to.
        examples: The sentences illustrating it.
    """

    glosses: list[str]
    tags: list[str]
    topics: list[str]
    examples: list[_RawExample]


class _RawForm(TypedDict, total=False):
    """
    One written form of an entry, inflected or otherwise.

    Attributes:
        form: The form itself.
        tags: What it is a form of, and how it was arrived at.
    """

    form: str
    tags: list[str]


class _RawEntry(TypedDict, total=False):
    """
    One dictionary entry.

    Attributes:
        word: The headword.
        pos: Its part of speech.
        lang_code: The language the headword belongs to.
        forms: The shapes the headword takes.
        senses: Its meanings.
    """

    word: str
    pos: str
    lang_code: str
    forms: list[_RawForm]
    senses: list[_RawSense]


class WiktionaryExtractor:
    """
    Reads lemmas out of what wiktextract made of a Wiktionary dump.

    The filters are settled once, on the extractor, since every level of an
    entry reads them.
    """

    def __init__(
        self,
        language: str,
        allowed_pos: frozenset[POS] | None,
        minimum_year: int | None,
        maximum_year: int | None,
    ) -> None:
        """
        Set the filters every extraction will answer to.

        Args:
            language: Wiktionary's code for the language to read, such as en.
            allowed_pos: Parts of speech to keep, or None for every known one.
            minimum_year: Oldest quotation to keep, or None for no bound.
            maximum_year: Newest quotation to keep, or None for no bound.
        """
        self._language: str = language

        self._allowed_pos: frozenset[POS] | None = allowed_pos

        self._minimum_year: int | None = minimum_year
        self._maximum_year: int | None = maximum_year

    @staticmethod
    def _open(
        input_path: Path,
    ) -> IO[str]:
        """
        Open a wiktextract file, decompressing it if need be.

        Args:
            input_path: The file to read.

        Returns:
            The open file, in text mode.
        """
        match input_path.suffix:
            case ".zst":
                return zstd.open(input_path, "rt", encoding="utf-8")

            case ".gz":
                return gzip.open(input_path, "rt", encoding="utf-8")

            case _:
                return input_path.open(encoding="utf-8")

    @staticmethod
    def _parse_year(
        reference: str,
    ) -> int | None:
        """
        Read the year of publication off a reference.

        Args:
            reference: The source, as Wiktionary formats it.

        Returns:
            The first year the reference names, or None if it names none.
        """
        match = _YEAR_PATTERN.search(reference)

        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_forms(
        raw_forms: list[_RawForm],
        lemma: str,
    ) -> frozenset[str]:
        """
        Collect the shapes an occurrence of the lemma may take.

        Args:
            raw_forms: What wiktextract listed under the entry.
            lemma: The headword, which is a form of itself.

        Returns:
            The headword and every inflection worth looking for.
        """
        forms = {lemma}

        for raw_form in raw_forms:
            form = raw_form.get("form", "").strip()
            if not form or form == _EMPTY_CELL:
                continue

            if not _SERVICE_TAGS.isdisjoint(raw_form.get("tags", [])):
                continue

            forms.add(form)

        return frozenset(forms)

    def _parse_sentences(
        self,
        raw_examples: list[_RawExample],
        forms: frozenset[str],
    ) -> list[Sentence]:
        """
        Collect the sentences illustrating one sense.

        The year filter reaches quotations alone: examples carry no reference,
        and so no date.

        Args:
            raw_examples: What wiktextract listed under the sense.
            forms: The shapes the lemma takes, to be located in each sentence.

        Returns:
            The sentences that survive it, in the order they were listed.
        """
        sentences: list[Sentence] = []

        for raw_example in raw_examples:
            text = raw_example.get("text", "").strip()
            if not text:
                continue

            word_offsets = find_word_offsets(text, forms)

            reference = raw_example.get("ref", "").strip()
            if not reference:
                sentences.append(
                    Example(
                        text,
                        word_offsets=word_offsets,
                    )
                )

                continue

            year = self._parse_year(reference)

            # An undated quotation only stands in the way once a bound is set.
            if self._minimum_year is not None or self._maximum_year is not None:
                if year is None:
                    continue

                if self._minimum_year is not None and year < self._minimum_year:
                    continue

                if self._maximum_year is not None and year > self._maximum_year:
                    continue

            sentences.append(
                Quotation(
                    text,
                    reference,
                    year,
                    word_offsets=word_offsets,
                )
            )

        return sentences

    def _parse_senses(
        self,
        raw_senses: list[_RawSense],
        lemma_id: str,
        forms: frozenset[str],
    ) -> list[Sense]:
        """
        Collect the senses of one entry.

        A nested sense is kept alongside its parent rather than in its place,
        since a parent often carries examples of its own.

        Args:
            raw_senses: What wiktextract listed under the entry.
            lemma_id: Identifies the entry, and opens each sense identifier.
            forms: The shapes the lemma takes, to be located in each sentence.

        Returns:
            The senses that carry at least one gloss.
        """
        senses: list[Sense] = []

        for raw_sense in raw_senses:
            glosses = tuple(
                gloss.strip() for gloss in raw_sense.get("glosses", []) if gloss.strip()
            )
            if not glosses:
                continue

            senses.append(
                Sense(
                    f"{lemma_id}.{len(senses) + 1:02d}",
                    glosses,
                    tuple(raw_sense.get("topics", [])),
                    tuple(raw_sense.get("tags", [])),
                    self._parse_sentences(raw_sense.get("examples", []), forms),
                )
            )

        return senses

    def extract(
        self,
        input_path: Path,
    ) -> Iterator[Lemma]:
        """
        Read lemmas from a wiktextract file, one entry at a time.

        Entries sharing a lemma and a part of speech are left apart, since
        Wiktionary separates them for a reason: different etymologies. An
        ordinal tells them apart, Wiktionary's own numbers being too coarse.

        Identifiers read as bank.noun.2 for an entry and bank.noun.2.03 for a
        sense. Being positional, they hold only for the dump they came from.

        Args:
            input_path: The wiktextract file to read, compressed or not.

        Yields:
            One lemma per entry with a part of speech we keep and a sense.
        """
        ordinals: dict[str, int] = {}

        with self._open(input_path) as file:
            for line in tqdm(file, desc=input_path.name, unit=" entry"):
                try:
                    entry = cast(_RawEntry, json.loads(line))
                except json.JSONDecodeError:
                    # wiktextract carries its reporting among the entries.
                    continue

                # A dump holds every language Wiktionary describes.
                if entry.get("lang_code") != self._language:
                    continue

                lemma = entry.get("word", "").strip()
                if not lemma:
                    continue

                try:
                    pos = POS(entry.get("pos", ""))
                except ValueError:
                    # Wiktionary knows far more parts of speech than are kept.
                    continue

                if self._allowed_pos is not None and pos not in self._allowed_pos:
                    continue

                key = f"{lemma}.{pos}"
                ordinal = ordinals.get(key, 0) + 1

                lemma_id = f"{key}.{ordinal}"

                forms = self._parse_forms(entry.get("forms", []), lemma)

                senses = self._parse_senses(entry.get("senses", []), lemma_id, forms)
                if not senses:
                    continue

                # Spent only once the entry is kept.
                ordinals[key] = ordinal

                yield Lemma(
                    lemma_id,
                    lemma,
                    pos,
                    senses,
                )
