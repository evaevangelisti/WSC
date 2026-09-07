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
        Prompt templates shared by inference and experiments.
    """
    with path.open("rb") as stream:
        records = cast(dict[str, PromptRecord], tomllib.load(stream))

    return AlignmentPrompts(
        path.stem,
        {task: record["template"] for task, record in records.items()},
    )
