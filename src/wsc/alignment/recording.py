"""Record alignment decisions incrementally in resource-specific tables."""

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

from ..constants import ALIGNMENT_FIELDS
from ..export import TSVWriter
from ..files import partial_file
from ..models.alignment import AlignmentResult, AlignmentTask


def serialize_alignment(
    result: AlignmentResult,
) -> Iterator[dict[str, str]]:
    """
    Serialize accepted links and abstentions with their source context.

    Args:
        result: Validated model decisions.

    Yields:
        One row per accepted link or explicit abstention.
    """
    for decision in result.decisions:
        for link in decision.links or (None,):
            yield {
                "alignment_id": result.query.alignment_id,
                "source_id": decision.source_id,
                "target_id": link.target_id if link else "",
                "relation": link.relation if link else "",
                "reason": link.reason if link else "",
            }


def open_alignment_recorder(
    stack: ExitStack,
    paths: dict[AlignmentTask, Path],
    metadata: dict[str, object],
) -> Callable[[AlignmentResult], None]:
    """
    Open task-specific writers and return their decision recorder.

    Args:
        stack: Context managing atomic output writers.
        paths: Destination for each requested task.
        metadata: Model settings and input fingerprints.

    Returns:
        A callback persisting each alignment result.
    """
    metadata_path = next(iter(paths.values())).with_name("metadata.json")

    metadata_partial = stack.enter_context(partial_file(metadata_path))
    _ = metadata_partial.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    writers = {
        task: stack.enter_context(TSVWriter(path, ALIGNMENT_FIELDS))
        for task, path in paths.items()
    }

    def record_decisions(
        result: AlignmentResult,
    ) -> None:
        """
        Persist decisions in the corresponding task table.

        Args:
            result: Complete alignment result.
        """
        for row in serialize_alignment(result):
            writers[result.query.task].write(row)

        writers[result.query.task].flush()

    return record_decisions
