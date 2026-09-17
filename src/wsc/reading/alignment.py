"""Read alignment queries and persisted language model decisions."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from itertools import groupby
from pathlib import Path
from typing import NotRequired, TypedDict

from ..models import POS
from ..models.alignment import (
    AlignmentDecision,
    AlignmentLink,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    Definition,
)


class DefinitionRecord(TypedDict):
    """Represent a serialized lexical definition."""

    id: str
    glosses: list[str]
    synonyms: NotRequired[list[str]]


class QueryRecord(TypedDict):
    """Represent an alignment query in caches and annotation exports."""

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
    Restore definitions from a stored alignment query.

    Args:
        record: Serialized source and candidate context.

    Returns:
        Complete definitions with stable identifiers.
    """
    return AlignmentQuery(
        AlignmentTask(record["task"]),
        record["alignment_id"],
        record["lemma_id"],
        record["lemma"],
        POS(record["pos"]),
        tuple(
            Definition(
                item["id"],
                tuple(item["glosses"]),
                tuple(item.get("synonyms", ())),
            )
            for item in record["source_definitions"]
        ),
        tuple(
            Definition(
                item["id"],
                tuple(item["glosses"]),
                tuple(item.get("synonyms", ())),
            )
            for item in record["target_definitions"]
        ),
    )


def read_alignment_cache(
    path: Path,
) -> dict[str, dict[str, AlignmentDecision]]:
    """
    Index persisted decisions by query and source identifiers.

    Args:
        path: Task-specific alignment TSV.

    Returns:
        Source decisions grouped by alignment identifier.

    Raises:
        ValueError: If the table schema or row structure is invalid.
    """
    if not path.is_file():
        return {}

    from ..constants import ALIGNMENT_FIELDS

    cache: dict[str, dict[str, AlignmentDecision]] = {}

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")

        if tuple(reader.fieldnames or ()) != ALIGNMENT_FIELDS:
            raise ValueError(f"Invalid alignment table fields: {path}")

        for alignment_id, rows in groupby(reader, key=lambda row: row["alignment_id"]):
            if not alignment_id or alignment_id in cache:
                raise ValueError(f"Repeated alignment record: {alignment_id!r}")

            decisions: dict[str, AlignmentDecision] = {}

            for source_id, source_rows in groupby(
                rows,
                key=lambda row: row["source_id"],
            ):
                records = list(source_rows)

                if not source_id or source_id in decisions:
                    raise ValueError(f"Repeated source decision: {source_id!r}")

                for row in records:
                    association = (
                        bool(row["target_id"]),
                        bool(row["relation"]),
                        bool(row["reason"]),
                    )

                    if any(association) != all(association):
                        raise ValueError(f"Incomplete source decision: {source_id}")

                decisions[source_id] = AlignmentDecision(
                    source_id,
                    tuple(
                        AlignmentLink(
                            source_id,
                            row["target_id"],
                            row["relation"],
                            row["reason"],
                        )
                        for row in records
                        if row["target_id"]
                    ),
                )

            cache[alignment_id] = decisions

    return cache


def read_alignments(
    path: Path,
    queries: Mapping[str, AlignmentQuery],
) -> Iterator[AlignmentResult]:
    """
    Stream validated decisions from an alignment table.

    Args:
        path: Completed language model alignment TSV.
        queries: Queries indexed by alignment identifier.

    Yields:
        Decisions grouped by their original query.

    Raises:
        ValueError: If the cache schema or decisions are incompatible.
    """
    from ..alignment.decisions import validate_result

    for alignment_id, decisions in read_alignment_cache(path).items():
        try:
            query = queries[alignment_id]
        except KeyError as error:
            raise ValueError(f"Unknown cached alignment: {alignment_id}") from error

        result = AlignmentResult(query, tuple(decisions.values()))
        validate_result(result)

        yield result
