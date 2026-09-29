"""
Read generic synsets supplied as JSON Lines.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..files import open_compressed
from ..models import POS, Synset, SynsetResource


class SynsetResourceRecord(TypedDict):
    """
    Serialized lexical evidence from one resource.
    """

    members: list[str]
    glosses: list[str]
    examples: NotRequired[list[str]]


class SynsetRecord(TypedDict):
    """
    Serialized generic synset accepted by the align command.
    """

    id: NotRequired[str]
    pos: str
    sources: dict[str, SynsetResourceRecord]


def read_synsets(
    path: Path,
) -> Iterator[Synset]:
    """
    Stream generic synsets from a JSON Lines file.

    Args:
        path: Input JSON Lines file.

    Yields:
        Validated synsets with generated identifiers where absent.

    Raises:
        ValueError: If a record has no resource or a resource lacks evidence.
    """
    with open_compressed(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = cast(SynsetRecord, json.loads(line))

            resources = {
                name: SynsetResource(
                    tuple(source["members"]),
                    tuple(source["glosses"]),
                    tuple(source.get("examples", ())),
                )
                for name, source in record["sources"].items()
            }

            if not resources:
                raise ValueError(f"Synset line {line_number} needs members and glosses")

            if any(not name or name == "remaining" for name in resources):
                raise ValueError(f"Synset line {line_number} has invalid source names")

            if any(
                not resource.members or not resource.glosses
                for resource in resources.values()
            ):
                raise ValueError(f"Synset line {line_number} needs members and glosses")

            yield Synset(
                record.get("id", f"synset-{line_number:08d}"),
                POS(record["pos"]),
                resources,
            )
