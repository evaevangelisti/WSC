"""Read alignment queries and persisted language model decisions."""

import csv
import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import NotRequired, TypedDict, cast

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


def read_metadata(
    path: Path,
) -> dict[str, str]:
    """
    Read inference settings from a cached alignment table.

    Args:
        path: Alignment TSV file.

    Returns:
        Persisted inference settings.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream, delimiter="\t"))

    return cast(dict[str, str], json.loads(row["context"]))


def read_alignments(
    path: Path,
) -> Iterator[AlignmentResult]:
    """
    Stream validated decisions from an alignment table.

    Args:
        path: Completed language model alignment TSV.

    Yields:
        Decisions grouped by their original query.

    Raises:
        ValueError: If the cache schema or decisions are incompatible.
    """
    from ..alignment.decisions import validate_result
    from ..constants import ALIGNMENT_SCHEMA

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")

        metadata = cast(dict[str, str], json.loads(next(reader)["context"]))
        if metadata.get("schema") != ALIGNMENT_SCHEMA:
            raise ValueError("Alignment cache requires the language model schema")

        for alignment_id, rows in groupby(reader, key=lambda row: row["alignment_id"]):
            records = list(rows)
            first = records[0]

            query = parse_query(cast(QueryRecord, json.loads(first["context"])))
            if query.alignment_id != alignment_id:
                raise ValueError(f"Evidence identity differs: {alignment_id}")

            decisions: list[AlignmentDecision] = []

            for source_id, source_rows in groupby(
                records,
                key=lambda row: row["source_id"],
            ):
                if not source_id:
                    continue

                source_records = list(source_rows)

                decisions.append(
                    AlignmentDecision(
                        source_id,
                        tuple(
                            AlignmentLink(
                                source_id,
                                row["target_id"],
                                row["relation"],
                                row["reason"],
                            )
                            for row in source_records
                            if row["target_id"]
                        ),
                    )
                )

            result = AlignmentResult(query, tuple(decisions), first["response"])
            validate_result(result)

            yield result
