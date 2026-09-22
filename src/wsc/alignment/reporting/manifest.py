"""Describe the sources and settings used to align collected senses."""

from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from ...models.alignment import (
    AlignmentPrompts,
    AlignmentTask,
    GlossMode,
    ModelSettings,
)
from ...reporting import describe_source


def build_manifest(
    input_path: Path,
    settings: ModelSettings,
    mode: GlossMode,
    prompts: AlignmentPrompts,
    tasks: tuple[AlignmentTask, ...],
) -> dict[str, object]:
    """
    Describe the source and configuration of an alignment run.

    Args:
        input_path: Collected senses supplied to alignment.
        settings: Language model generation settings.
        mode: Wiktionary definition representation.
        prompts: Selected prompt collection.
        tasks: Requested alignment resources.

    Returns:
        Provenance completed after alignment succeeds.
    """
    return {
        "source": describe_source(input_path),
        "tasks": list(tasks),
        "settings": {
            **asdict(settings),
            "gloss_mode": mode,
            "prompts": prompts.name,
        },
        "runtime": {
            "python": python_version(),
            "packages": {"wsc": version("wsc")},
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
