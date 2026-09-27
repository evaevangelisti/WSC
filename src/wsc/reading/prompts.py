"""
Read task-specific alignment prompts from TOML.
"""

import tomllib
from pathlib import Path
from typing import TypedDict, cast

from ..files import COMPRESSED_SUFFIXES, open_compressed
from ..models.alignment import AlignmentPrompts


class PromptRecord(TypedDict):
    """
    Represent serialized task prompts.
    """

    template: str


def read_prompts(
    path: Path,
) -> AlignmentPrompts:
    """
    Load one prompt template for each task.

    Args:
        path: TOML file containing task templates.

    Returns:
        Prompt templates used by alignment inference.
    """
    with open_compressed(path, "rb") as stream:
        records = cast(dict[str, object], tomllib.load(stream))

    name = (
        path.with_suffix("").stem if path.suffix in COMPRESSED_SUFFIXES else path.stem
    )

    return AlignmentPrompts(
        name,
        cast(str, records["system"]),
        {
            task: cast(PromptRecord, record)["template"]
            for task, record in records.items()
            if task != "system"
        },
    )
