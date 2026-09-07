"""
Read named semantic instruction profiles from TOML.
"""

import tomllib
from pathlib import Path
from typing import TypedDict, cast

from ..models.alignment import AlignmentInstructions


class InstructionRecord(TypedDict):
    """Serialized instruction and its relation hypotheses."""

    instruction: str
    relations: dict[str, str]


def read_instructions(
    path: Path,
) -> tuple[AlignmentInstructions, ...]:
    """
    Load profiles in their declared order without altering instruction text.

    Args:
        path: TOML file containing named profile tables.

    Returns:
        Profiles shared by model inference, scoring, and provenance.
    """
    with path.open("rb") as stream:
        records = cast(dict[str, InstructionRecord], tomllib.load(stream))

    return tuple(
        AlignmentInstructions(name, record["instruction"], record["relations"])
        for name, record in records.items()
    )
