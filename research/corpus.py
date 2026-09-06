"""
Reading the export the studies draw from.

The schema below is the one the JSONL writer produces, declared once so that
every study reads it the same way. A key the writer prunes is optional.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

DATA = Path(__file__).resolve().parent.parent / "data"
"""Where the exports the studies draw from are kept, outside version control."""

SENSES = DATA / "senses.jsonl"
"""The collected senses, as `wsc collect` writes them."""

SEPARATOR = " > "
"""What joins a gloss chain, Wiktionary nesting its senses."""


class WordOffset(TypedDict):
    """
    One candidate range and the methods supporting it.

    Attributes:
        offset: Half-open code-point range.
        sources: Methods supporting the candidate.
    """

    offset: list[int]
    sources: list[Literal["bold", "lemmatizer"]]


class Sentence(TypedDict):
    """
    A sentence Wiktionary hangs off a sense.

    A reference makes it a quotation, and its absence an example. Offsets are
    left out where the headword is not spelled in the sentence.
    """

    text: str
    word_offsets: NotRequired[list[WordOffset]]
    reference: NotRequired[str]
    year: NotRequired[int]


class Sense(TypedDict):
    """
    One meaning of an entry, with the sentences illustrating it.
    """

    id: str
    glosses: list[str]
    etymology: NotRequired[str]
    synonyms: NotRequired[list[str]]
    topics: NotRequired[list[str]]
    tags: NotRequired[list[str]]
    sentences: NotRequired[list[Sentence]]
    sense_ids: NotRequired[list[str]]
    wikidata_ids: NotRequired[list[str]]


class Entry(TypedDict):
    """
    One lemma, holding every sense it carries and what it is called elsewhere.
    """

    id: str
    lemma: str
    pos: str
    variants: NotRequired[list[str]]
    senses: NotRequired[list[Sense]]
    translations: NotRequired[dict[str, dict[str, list[str]]]]


def read(
    path: Path = SENSES,
) -> Iterator[Entry]:
    """
    Read an export written one JSON object per line.

    Args:
        path: The file to read.

    Yields:
        One entry per line.

    Raises:
        SystemExit: If the export is not there, since every study needs it.
    """
    if not path.exists():
        raise SystemExit(f"No export at {path}; collect one there first")

    with path.open(encoding="utf-8") as file:
        for line in file:
            # A decoder knows nothing of what it decodes. The export comes out
            # of the writer and is read back against its own models.
            yield cast(Entry, json.loads(line))
