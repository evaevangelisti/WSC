"""Read generic synsets supplied as JSON Lines."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..models import POS, Synset, SynsetMember


class SynsetRecord(TypedDict):
    """Serialized generic synset accepted by the align command."""

    id: NotRequired[str]
    pos: str
    members: list[str] | dict[str, list[str]]
    glosses: list[str]
    examples: NotRequired[list[str]]


def _read_members(
    members: list[str] | dict[str, list[str]],
) -> tuple[SynsetMember, ...]:
    """Flatten members while retaining any source declared for them.

    Args:
        members: Plain members or members grouped by their source.

    Returns:
        Members in their input order.
    """
    if isinstance(members, list):
        return tuple(SynsetMember(member) for member in members)

    return tuple(
        SynsetMember(member, source)
        for source, source_members in members.items()
        for member in source_members
    )


def read_synsets(
    path: Path,
) -> Iterator[Synset]:
    """Stream generic synsets from a JSON Lines file.

    Args:
        path: Input JSON Lines file.

    Yields:
        Validated synsets with generated identifiers where absent.

    Raises:
        ValueError: If a record has no members or no glosses.
    """
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = cast(SynsetRecord, json.loads(line))

            members = _read_members(record["members"])
            glosses = tuple(record["glosses"])

            if not members or not glosses:
                raise ValueError(f"Synset line {line_number} needs members and glosses")

            yield Synset(
                record.get("id", f"synset-{line_number:08d}"),
                POS(record["pos"]),
                members,
                glosses,
                tuple(record.get("examples", [])),
            )
