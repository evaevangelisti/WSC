"""Record alignment decisions incrementally in resource-specific tables."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

from ..constants import ALIGNMENT_FIELDS
from ..export import TSVWriter
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
) -> Callable[[AlignmentResult], None]:
    """
    Open task-specific writers and return their decision recorder.

    Args:
        stack: Context managing atomic output writers.
        paths: Destination for each requested task.

    Returns:
        A callback persisting each alignment result.
    """
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
