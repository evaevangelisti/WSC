"""Read alignment queries and persisted language model decisions."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from ..models import POS
from ..models.alignment import (
    AlignmentDecision,
    AlignmentLink,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    Definition,
)

if TYPE_CHECKING:
    from ..alignment.candidates import WordNetCandidates


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
    return cast(
        dict[str, str],
        json.loads(path.with_name("metadata.json").read_text(encoding="utf-8")),
    )


def read_queries(
    input_path: Path,
    tasks: tuple[AlignmentTask, ...],
    candidates: WordNetCandidates,
) -> dict[str, AlignmentQuery]:
    """
    Rebuild alignment queries needed to replay cached decisions.

    Args:
        input_path: Collected entries used for the original alignment.
        tasks: Alignment resources represented by the cache.
        candidates: WordNet candidate index.

    Returns:
        Queries indexed by their stable alignment identifier.
    """
    from ..alignment.tasks import build_queries
    from .wiktionary import read_lemmas

    queries: dict[str, AlignmentQuery] = {}

    for lemma in read_lemmas(input_path):
        for task in tasks:
            for query in build_queries(lemma, task, candidates):
                queries[query.alignment_id] = query

    return queries


def read_alignments(
    path: Path,
    queries: dict[str, AlignmentQuery],
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

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")

        for alignment_id, rows in groupby(reader, key=lambda row: row["alignment_id"]):
            records = list(rows)

            try:
                query = queries[alignment_id]
            except KeyError as error:
                raise ValueError(f"Unknown cached alignment: {alignment_id}") from error

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

            result = AlignmentResult(query, tuple(decisions))
            validate_result(result)

            yield result
