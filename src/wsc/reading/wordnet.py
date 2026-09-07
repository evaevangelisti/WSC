"""Read extracted WordNet concepts from the source cache."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..models import POS, Synset


class SynsetRecord(TypedDict):
    """Serialized WordNet concept from the WSC cache."""

    id: str
    ili: str
    pos: str
    definition: str
    members: NotRequired[list[str]]
    hypernyms: NotRequired[list[str]]
    examples: NotRequired[list[str]]


def read_synsets(
    path: Path,
) -> Iterator[Synset]:
    """
    Stream WordNet concepts from their extracted cache.

    Args:
        path: Cached synset JSONL file.

    Yields:
        WordNet concepts with cross-edition identifiers.
    """
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = cast(SynsetRecord, json.loads(line))

            yield Synset(
                record["id"],
                record["ili"],
                POS(record["pos"]),
                record["definition"],
                tuple(record.get("members", [])),
                tuple(record.get("hypernyms", [])),
                tuple(record.get("examples", [])),
            )
