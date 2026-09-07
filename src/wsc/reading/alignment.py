"""
Read alignment queries and persisted TSV evidence.
"""

import csv
import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import TypedDict, cast

from ..models import POS
from ..models.alignment import (
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Definition,
)


class DefinitionRecord(TypedDict):
    """Serialized candidate definition."""

    id: str
    glosses: list[str]


class QueryRecord(TypedDict):
    """Serialized alignment input retained alongside scores and annotations."""

    task: str
    alignment_id: str
    lemma_id: str
    lemma: str
    pos: str
    source_definitions: list[DefinitionRecord]
    target_definitions: list[DefinitionRecord]


def parse_query(
    record: QueryRecord,
) -> AlignmentQuery:
    """
    Restore persisted candidate definitions.

    Args:
        record: Serialized alignment query.

    Returns:
        Complete candidate sets with stable identifiers.
    """
    return AlignmentQuery(
        AlignmentTask(record["task"]),
        record["alignment_id"],
        record["lemma_id"],
        record["lemma"],
        POS(record["pos"]),
        tuple(
            Definition(item["id"], tuple(item["glosses"]))
            for item in record["source_definitions"]
        ),
        tuple(
            Definition(item["id"], tuple(item["glosses"]))
            for item in record["target_definitions"]
        ),
    )


def read_metadata(
    path: Path,
) -> dict[str, str]:
    """
    Read inference settings without loading candidate scores.

    Args:
        path: Cached alignment TSV.

    Returns:
        Settings persisted with the inference run.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))

    return cast(dict[str, str], json.loads(row["context"]))


def read_alignments(
    path: Path,
) -> Iterator[AlignmentResult]:
    """
    Stream candidate evidence from a resource-specific TSV.

    Args:
        path: Completed cached alignment file.

    Yields:
        Candidate scores grouped by original query.

    Raises:
        ValueError: If an evidence row has inconsistent query identity.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")

        _ = next(reader)

        for alignment_id, rows in groupby(reader, key=lambda row: row["alignment_id"]):
            first = next(rows)

            query = parse_query(cast(QueryRecord, json.loads(first["context"])))
            if query.alignment_id != alignment_id:
                raise ValueError(f"Evidence identity differs: {alignment_id}")

            scores = tuple(
                AlignmentScore(
                    row["source_id"],
                    row["target_id"],
                    row["relation"],
                    float(row["score"]),
                )
                for row in (first, *rows)
                if row["source_id"]
            )

            yield AlignmentResult(query, scores)
