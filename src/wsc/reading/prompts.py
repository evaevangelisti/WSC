"""Read task-specific alignment prompts from TOML."""

import tomllib
from pathlib import Path
from typing import TypedDict, cast

from ..models.alignment import AlignmentPrompts


class PromptRecord(TypedDict):
    """Represent serialized task prompts."""

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
    with path.open("rb") as stream:
        records = cast(dict[str, object], tomllib.load(stream))

    return AlignmentPrompts(
        path.stem,
        cast(str, records["system"]),
        {
            task: cast(PromptRecord, record)["template"]
            for task, record in records.items()
            if task != "system"
        },
    )
