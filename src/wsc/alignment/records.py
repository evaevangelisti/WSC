"""
Serialize alignment records and manage their resource-specific outputs.
"""

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from ..constants import ALIGNMENT_FIELDS
from ..export import TSVWriter
from ..models.alignment import AlignmentResult, AlignmentTask


def serialize_alignment(
    result: AlignmentResult,
) -> Iterator[dict[str, str]]:
    """
    Serialize candidate scores while retaining queries without candidates.

    Args:
        result: Complete evidence for one alignment query.

    Yields:
        Tabular records with definitions stored on the first query row.
    """
    context = json.dumps(asdict(result.query), ensure_ascii=False)

    if not result.scores:
        yield {
            "alignment_id": result.query.alignment_id,
            "context": context,
        }

    for position, score in enumerate(result.scores):
        yield {
            "alignment_id": result.query.alignment_id,
            "source_id": score.source_id,
            "target_id": score.target_id,
            "relation": score.relation,
            "score": repr(score.score),
            "context": context if position == 0 else "",
        }


def open_alignment_recorder(
    stack: ExitStack,
    paths: dict[AlignmentTask, Path],
    metadata: dict[str, str],
) -> Callable[[AlignmentResult], None]:
    """
    Open resource-specific tables and prepare their score recorder.

    Args:
        stack: Context owning the atomic output writers.
        paths: Destination for each requested alignment task.
        metadata: Inference settings persisted before candidate scores.

    Returns:
        Callback serializing scores into the corresponding resource table.
    """
    writers = {
        task: stack.enter_context(TSVWriter(path, ALIGNMENT_FIELDS))
        for task, path in paths.items()
    }
    for writer in writers.values():
        writer.write({"context": json.dumps(metadata, ensure_ascii=False)})

    def record_scores(
        result: AlignmentResult,
    ) -> None:
        """
        Persist scores under their resource.

        Args:
            result: Complete candidate evidence.
        """
        for row in serialize_alignment(result):
            writers[result.query.task].write(row)

    return record_scores
