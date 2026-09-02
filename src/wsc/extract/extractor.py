"""
Extraction of lemmas from wiktextract output.
"""

import gzip
import json
from collections.abc import Iterator, Sequence
from compression import zstd
from dataclasses import replace
from pathlib import Path
from typing import IO, cast

from kwic import Locator, Query
from tqdm import tqdm

from ..models import POS, Lemma
from .identifiers import lemma_id
from .offsets import build_query, find_word_offsets
from .parts import (
    Variants,
    alternative_forms,
    gather_variants,
    parse_forms,
    parse_senses,
    parse_translations,
)
from .schema import RawEntry

# What the bar says while the other spellings of a headword are gathered,
# and while the sentences are read.
_GATHERING_VARIANTS = "Gathering variants"
_LOCATING = "Locating the lemmas"


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
        locator: Locator,
    ) -> None:
        """
        Set the filters every extraction will answer to.

        Args:
            language: Wiktionary's code for the language to read, such as en.
            allowed_pos: Parts of speech to keep, or None for every known one.
            minimum_year: Oldest quotation to keep, or None for no bound.
            maximum_year: Newest quotation to keep, or None for no bound.
            locator: The search the lemma is located with.
        """
        self._language: str = language

        self._allowed_pos: frozenset[POS] | None = allowed_pos

        self._minimum_year: int | None = minimum_year
        self._maximum_year: int | None = maximum_year

        self._locator: Locator = locator

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

    def _read_entries(
        self,
        input_path: Path,
        description: str,
    ) -> Iterator[RawEntry]:
        """
        Walk the entries of a wiktextract file, passing over what is not one.

        Args:
            input_path: The wiktextract file to read, compressed or not.
            description: What the progress bar says the pass is doing.

        Yields:
            One entry per line that holds a whole one.
        """
        with self._open(input_path) as file:
            for line in tqdm(file, desc=description, unit=" entry"):
                # wiktextract carries its reporting among the entries.
                if not line.startswith("{"):
                    continue

                try:
                    yield cast(RawEntry, json.loads(line))
                except json.JSONDecodeError:
                    # A dump cut short leaves an entry that opens and no more.
                    continue

    def _parse_entry(
        self,
        entry: RawEntry,
        lemma: str,
        pos: POS,
        variants: Variants,
    ) -> Lemma | None:
        """
        Read one entry into a lemma, or into nothing where it defines none.

        The entry is named after the meanings it holds, so that a page
        reordered upstream reads back under the identifier it had.

        Args:
            entry: The entry, as wiktextract wrote it.
            lemma: The headword.
            pos: Its part of speech.
            variants: How else each headword is written, read off the
            entries pointing at it.

        Returns:
            The lemma, or None where no sense of it survived the filters.
        """
        key = f"{lemma}.{pos}"

        senses = parse_senses(
            entry.get("senses", []),
            key,
            self._minimum_year,
            self._maximum_year,
        )
        if not senses:
            return None

        return Lemma(
            lemma_id(key, (sense.id for sense in senses)),
            lemma,
            pos,
            variants.get((lemma, pos), frozenset())
            | alternative_forms(entry.get("forms", []), lemma),
            senses,
            parse_translations(entry.get("translations", [])),
        )

    def _read_lemmas(
        self,
        input_path: Path,
        variants: Variants,
    ) -> Iterator[tuple[Lemma, Query]]:
        """
        Read the lemmas a wiktextract file holds, one entry at a time.

        Args:
            input_path: The wiktextract file to read, compressed or not.
            variants: How else each headword is written.

        Yields:
            One lemma per entry with a part of speech we keep and a sense,
            and what its sentences are to be searched for.
        """
        for entry in self._read_entries(input_path, input_path.name):
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

            parsed_entry = self._parse_entry(entry, lemma, pos, variants)
            if parsed_entry is None:
                continue

            forms = parse_forms(entry.get("forms", []), lemma)

            yield parsed_entry, build_query(lemma, pos, forms)

    def _locate(
        self,
        queried_lemmas: Sequence[tuple[Lemma, Query]],
    ) -> Iterator[Lemma]:
        """
        Say where the lemma falls in every sentence of every entry read.

        Every sentence goes to the search at once, so the model reads one long
        stream and pipes it as it sees fit rather than an entry at a time.

        Args:
            queried_lemmas: The entries read, each with what to search its
            sentences for.

        Yields:
            The same entries, their sentences carrying the ranges the lemma
            occupies.
        """
        sentences = sum(
            len(sense.sentences)
            for lemma, _ in queried_lemmas
            for sense in lemma.senses
        )

        located_offsets = find_word_offsets(
            self._locator,
            (
                (sentence.text, query)
                for lemma, query in queried_lemmas
                for sense in lemma.senses
                for sentence in sense.sentences
            ),
        )

        # Held open, so that a run cut short leaves no bar behind it.
        with tqdm(total=sentences, desc=_LOCATING, unit=" sentence") as progress:
            for lemma, _ in queried_lemmas:
                for sense in lemma.senses:
                    # Walked in the order the searches went out, so a sense
                    # takes as many ranges off the stream as it has sentences.
                    sense.sentences = [
                        replace(sentence, word_offsets=word_offsets)
                        for sentence, word_offsets in zip(
                            sense.sentences, located_offsets, strict=False
                        )
                    ]

                    _ = progress.update(len(sense.sentences))

                yield lemma

    def extract(
        self,
        input_path: Path,
    ) -> Iterator[Lemma]:
        """
        Read the lemmas of a wiktextract file, located in what attests them.

        The file is walked twice: an inflection sits on a page of its own and
        points back at the lemma, which may be anywhere in it. Every entry is
        then held while the search reads the sentences of all of them.

        Args:
            input_path: The wiktextract file to read, compressed or not.

        Yields:
            One lemma per entry with a part of speech we keep and a sense.
        """
        variants = gather_variants(
            self._read_entries(input_path, _GATHERING_VARIANTS),
            self._language,
        )

        yield from self._locate(list(self._read_lemmas(input_path, variants)))
