"""
Extraction of lemmas from wiktextract output.
"""

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

from kwic import Locator, Query
from tqdm import tqdm

from ...constants import LANGUAGE
from ...models import POS, Lemma, Translations
from ..offsets import build_query, find_word_offsets
from .entries import read_entries
from .identifiers import lemma_id
from .merge import merge_lemmas, merge_word_offsets
from .parts import (
    Variants,
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
        allowed_pos: frozenset[POS] | None,
        minimum_year: int | None,
        maximum_year: int | None,
        locator: Locator,
        off_page_translations: dict[str, Translations] | None = None,
    ) -> None:
        """
        Set the filters every extraction will answer to.

        Args:
            allowed_pos: Parts of speech to keep, or None for every known one.
            minimum_year: Oldest quotation to keep, or None for no bound.
            maximum_year: Newest quotation to keep, or None for no bound.
            locator: The search the lemma is located with.
            off_page_translations: What each entry is translated by elsewhere, or None
            where the dump was not walked for it.
        """
        self._allowed_pos: frozenset[POS] | None = allowed_pos

        self._minimum_year: int | None = minimum_year
        self._maximum_year: int | None = maximum_year

        self._locator: Locator = locator

        self._off_page_translations: dict[str, Translations] = (
            off_page_translations or {}
        )

    def _parse_entry(
        self,
        entry: RawEntry,
        lemma: str,
        pos: POS,
        variants: Variants,
    ) -> Lemma | None:
        """
        Read one entry into a lemma, or into nothing where it defines none.

        Args:
            entry: The entry, as wiktextract wrote it.
            lemma: The headword.
            pos: Its part of speech.
            variants: Variant lemma identifiers from entries pointing at each headword.

        Returns:
            The lemma, or None where no sense of it survived the filters.
        """
        entry_id = lemma_id(lemma, pos)

        senses = parse_senses(
            entry.get("senses", []),
            entry_id,
            lemma,
            entry.get("etymology_number", ""),
            self._minimum_year,
            self._maximum_year,
        )
        if not senses:
            return None

        return Lemma(
            entry_id,
            lemma,
            pos,
            variants.get((lemma, pos), frozenset()),
            senses,
            parse_translations(
                entry.get("translations", []),
                self._off_page_translations.get(entry_id),
            ),
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
            variants: Variant lemma identifiers grouped by headword and part of speech.

        Yields:
            One lemma per entry with a part of speech we keep and a sense,
            and what its sentences are to be searched for.
        """
        for entry in read_entries(input_path, input_path.name):
            # A dump holds every language Wiktionary describes.
            if entry.get("lang_code") != LANGUAGE:
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
                        replace(
                            sentence,
                            word_offsets=merge_word_offsets(
                                sentence.word_offsets,
                                word_offsets,
                            ),
                        )
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

        Walked twice, then held while sentences are read.

        Args:
            input_path: The wiktextract file to read, compressed or not.

        Yields:
            One lemma per headword and part of speech we keep, carrying the
            senses of every etymology the page states.
        """
        variants = gather_variants(
            read_entries(
                input_path,
                _GATHERING_VARIANTS,
            ),
        )

        yield from self._locate(
            list(
                merge_lemmas(
                    self._read_lemmas(
                        input_path,
                        variants,
                    )
                )
            )
        )
