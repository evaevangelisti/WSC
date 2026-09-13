"""How else a lemma is spelled, gathered from the entries stating so."""

from collections import defaultdict
from collections.abc import Iterable, Iterator

from ....constants import LANGUAGE
from ....identifiers import lemma_id
from ....models import POS
from ..schema import RawEntry, RawSense

_ALT_OF = "alt-of"

type Variants = dict[tuple[str, POS], frozenset[str]]
"""Variant lemma identifiers grouped by headword and part of speech."""


def _read_pointed_lemmas(
    raw_sense: RawSense,
) -> Iterator[str]:
    """
    Read the lemmas one sense states its headword to be a spelling of.

    Args:
        raw_sense: What wiktextract listed under the entry.

    Yields:
        Each lemma pointed at, named and stripped.
    """
    for pointed in raw_sense.get("alt_of", []):
        word = pointed.get("word", "").strip()

        if word:
            yield word


def gather_variants(
    entries: Iterable[RawEntry],
) -> Variants:
    """
    Gather every headword that states itself to be a spelling of another.

    Wiktionary writes a spelling on a page of its own, pointing back at the lemma, so
    the two meet only once the whole file has been read.

    Args:
        entries: The wiktextract file, read whole.

    Returns:
        Variant lemma identifiers grouped by headword and part of speech.
    """
    variants: defaultdict[tuple[str, POS], set[str]] = defaultdict(set)

    for entry in entries:
        if entry.get("lang_code") != LANGUAGE:
            continue

        variant = entry.get("word", "").strip()

        if not variant:
            continue

        try:
            pos = POS(entry.get("pos", ""))
        except ValueError:
            continue

        for raw_sense in entry.get("senses", []):
            if _ALT_OF not in raw_sense.get("tags", []):
                continue

            for pointed_lemma in _read_pointed_lemmas(raw_sense):
                if pointed_lemma != variant:
                    variants[pointed_lemma, pos].add(lemma_id(variant, pos))

    return {
        pointed_lemma: frozenset(variant_ids)
        for pointed_lemma, variant_ids in variants.items()
    }
