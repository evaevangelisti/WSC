"""
Record alignment decisions incrementally in resource-specific tables.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import override

from ..constants import ALIGNMENT_FIELDS
from ..export import TSVWriter
from ..models.alignment import AlignmentDecision, AlignmentResult, AlignmentTask
from ..reading.alignment import read_alignment_cache

type AlignmentTableKey = AlignmentTask | tuple[AlignmentTask, str]
"""
Task key with an optional resource pass.
"""


def alignment_table_key(
    task: AlignmentTask,
    stage: str = "",
) -> AlignmentTableKey:
    """
    Distinguish each synset pass from ordinary task tables.

    Args:
        task: Alignment task owning the decisions.
        stage: Selected synset resource or remaining pass.

    Returns:
        Task key with an optional source-stage qualifier.
    """
    return (task, stage) if stage else task


def _serialize_decision(
    alignment_id: str,
    decision: AlignmentDecision,
) -> Iterator[dict[str, str]]:
    """
    Write one decision as association or abstention rows.

    Args:
        alignment_id: Query identifier owning the decision.
        decision: Source decision and its accepted links.

    Yields:
        One row per link or an empty-target abstention.
    """
    for link in decision.links or (None,):
        yield {
            "alignment_id": alignment_id,
            "source_id": decision.source_id,
            "target_id": link.target_id if link else "",
            "relation": link.relation if link else "",
            "reason": link.reason if link else "",
        }


class _AlignmentTSVWriter(TSVWriter):
    """
    Publish complete decisions even when inference stops early.
    """

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Merge completed decisions into the normal cache after failure.

        Args:
            exc_type: Exception type raised during alignment, if any.
            exc_value: Exception raised during alignment, if any.
            traceback: Traceback from the interrupted alignment, if any.
        """
        if exc_type is None:
            return super().__exit__(exc_type, exc_value, traceback)

        self._close()

        decisions = read_alignment_cache(self._output_path)

        for alignment_id, sources in read_alignment_cache(self._partial_path).items():
            decisions.setdefault(alignment_id, {}).update(sources)

        with TemporaryDirectory(
            prefix=".alignment-cache-",
            dir=self._output_path.parent,
        ) as directory:
            merged_path = Path(directory) / self._output_path.name

            with TSVWriter(merged_path, ALIGNMENT_FIELDS) as writer:
                for alignment_id, sources in decisions.items():
                    for decision in sources.values():
                        for row in _serialize_decision(alignment_id, decision):
                            writer.write(row)

            _ = merged_path.replace(self._output_path)

        self._partial_path.unlink()

        return None


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
        yield from _serialize_decision(result.query.alignment_id, decision)


def open_alignment_recorder(
    stack: ExitStack,
    paths: Mapping[AlignmentTableKey, Path],
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
        task: stack.enter_context(_AlignmentTSVWriter(path, ALIGNMENT_FIELDS))
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
            writers[alignment_table_key(result.query.task, result.query.stage)].write(
                row
            )

        writers[alignment_table_key(result.query.task, result.query.stage)].flush()

    return record_decisions
