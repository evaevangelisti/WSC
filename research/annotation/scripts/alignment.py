"""Model-blind alignment tasks and validated manual reference associations."""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from wsc.alignment import render_definition
from wsc.alignment.tasks import TASK_HANDLERS
from wsc.models.alignment import AlignmentQuery, AlignmentTask, GlossMode
from wsc.reading import QueryRecord, parse_query

from .sampling import repeat_items, split_queries, write_json

type Link = tuple[str, str, str]


class ResultValue(TypedDict):
    """Label Studio choices or free-text content."""

    choices: NotRequired[list[str]]
    text: NotRequired[list[str]]


class AnnotationResult(TypedDict):
    """One Label Studio control result."""

    from_name: str
    value: ResultValue


class Annotation(TypedDict):
    """One annotator's completed Label Studio judgement."""

    completed_by: int | str | dict[str, object]
    result: list[AnnotationResult]
    was_cancelled: NotRequired[bool]


class TaskData(TypedDict):
    """Task identity and original candidate context."""

    task: str
    alignment_id: str
    query: QueryRecord


class AnnotatedTask(TypedDict):
    """Label Studio task export with independent judgements."""

    data: TaskData
    annotations: list[Annotation]


@dataclass(frozen=True, slots=True)
class Judgement:
    """
    One manual alignment decision.

    Attributes:
        query: Original model-blind task.
        annotator: Label Studio annotator identity.
        status: Matched, absent, or uncertain annotation.
        links: Explicitly accepted directed associations.
        notes: Optional qualitative observations.
    """

    query: AlignmentQuery
    annotator: str
    status: str
    links: frozenset[Link]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoldSample:
    """
    Consolidated annotations without counting repeated tasks twice.

    Attributes:
        judgements: Agreed or adjudicated reference decisions.
        conflicts: Task identifiers needing adjudication.
        repeated: Tasks judged by multiple independent annotators.
        agreed: Repeated tasks agreeing before adjudication.
    """

    judgements: tuple[Judgement, ...]
    conflicts: tuple[str, ...]
    repeated: int
    agreed: int


def annotator_id(
    value: int | str | dict[str, object],
) -> str:
    """
    Identify the same annotator across compact and expanded Label Studio exports.

    Args:
        value: Annotator identifier or expanded profile.

    Returns:
        Stable annotator identifier independent of profile metadata.
    """
    return str(value["id"] if isinstance(value, dict) else value)


def build_task(
    query: AlignmentQuery,
) -> dict[str, object]:
    """
    Present every candidate link without revealing model predictions.

    Args:
        query: Complete source and target definitions.

    Returns:
        Label Studio task with dynamic choices and original context.
    """
    sources = "".join(
        (
            f"<li><b>S{position}</b>: "
            f"{escape(render_definition(source, GlossMode.FULL))}</li>"
        )
        for position, source in enumerate(query.source_definitions, 1)
    )
    targets = "".join(
        (
            f"<li><b>T{position}</b>: "
            f"{escape(render_definition(target, GlossMode.LAST))}</li>"
        )
        for position, target in enumerate(query.target_definitions, 1)
    )
    relations = TASK_HANDLERS[query.task].relations
    choices = [
        {
            "value": json.dumps([source.id, target.id, relation], ensure_ascii=False),
            "html": f"S{source_position} → T{target_position}: {relation}",
        }
        for source_position, source in enumerate(query.source_definitions, 1)
        for target_position, target in enumerate(query.target_definitions, 1)
        for relation in relations
    ]

    return {
        "data": {
            "task": query.task,
            "alignment_id": query.alignment_id,
            "target": f"{query.lemma} ({query.pos})",
            "definitions": (
                f"<h3>Wiktionary senses</h3><ul>{sources}</ul>"
                f"<h3>Candidates</h3><ul>{targets}</ul>"
            ),
            "choices": choices,
            "query": asdict(query),
        }
    }


def write_tasks(
    queries: list[AlignmentQuery],
    output_dir: Path,
    seed: int,
) -> None:
    """
    Write shared primary tasks and independent second assignments.

    Args:
        queries: Model-independent sample shared by every run.
        output_dir: Destination for task files and the frozen sample.
        seed: Repeated-task selection seed.
    """
    tasks = [build_task(query) for query in queries]
    write_json(output_dir / "tasks.json", tasks)
    write_json(output_dir / "tasks.second.json", repeat_items(tasks, seed))
    write_json(
        output_dir / "sample.json",
        {
            "seed": seed,
            "queries": [asdict(query) for query in queries],
            "splits": split_queries(queries, seed),
        },
    )


def _validate_judgement(
    judgement: Judgement,
) -> None:
    """
    Reject contradictory labels before they become reference data.

    Args:
        judgement: Manually selected links and decision.

    Raises:
        ValueError: If decisions violate candidate identity, relation, or cardinality.
    """
    query = judgement.query

    if judgement.status not in {
        "matched",
        "no_match",
        "uncertain",
    }:
        raise ValueError(
            f"Unknown judgement for {query.alignment_id}: {judgement.status}"
        )

    if (judgement.status == "matched" and not judgement.links) or (
        judgement.status in {"no_match", "uncertain"} and judgement.links
    ):
        raise ValueError(
            f"Matched status and selected links disagree: {query.alignment_id}"
        )

    relations = TASK_HANDLERS[query.task].relations
    sources = {source.id for source in query.source_definitions}
    targets = {target.id for target in query.target_definitions}

    for source, target, relation in judgement.links:
        if source not in sources or target not in targets or relation not in relations:
            raise ValueError(f"Unknown candidate or relation: {query.alignment_id}")

    if len({(source, target) for source, target, _ in judgement.links}) != len(
        judgement.links
    ):
        raise ValueError(f"Multiple relations for one pair: {query.alignment_id}")

    if query.task == AlignmentTask.TRANSLATIONS and (
        len({source for source, _, _ in judgement.links}) != len(judgement.links)
        or len({target for _, target, _ in judgement.links}) != len(judgement.links)
    ):
        raise ValueError(
            f"Translation judgement violates one-to-one alignment: {query.alignment_id}"
        )


def read_judgements(
    paths: Sequence[Path],
) -> list[Judgement]:
    """
    Import completed Label Studio annotations, including optional notes.

    Args:
        paths: Primary or secondary annotation exports.

    Returns:
        Validated independent judgements.

    Raises:
        ValueError: If required decisions or valid candidate links are missing.
    """
    judgements: list[Judgement] = []

    for path in paths:
        tasks = cast(list[AnnotatedTask], json.loads(path.read_text(encoding="utf-8")))

        for task in tasks:
            query = parse_query(task["data"]["query"])

            for annotation in task["annotations"]:
                if annotation.get("was_cancelled", False):
                    continue

                results = {
                    item["from_name"]: item["value"] for item in annotation["result"]
                }
                statuses = results.get("status", {}).get("choices", [])

                if len(statuses) != 1:
                    raise ValueError(
                        f"Exactly one status is required: {query.alignment_id}"
                    )

                values = [
                    cast(list[str], json.loads(value))
                    for value in results.get("links", {}).get("choices", [])
                ]

                if any(len(value) != 3 for value in values):
                    raise ValueError(
                        f"Incomplete source, target, or relation: {query.alignment_id}"
                    )

                judgement = Judgement(
                    query,
                    annotator_id(annotation["completed_by"]),
                    statuses[0],
                    frozenset((value[0], value[1], value[2]) for value in values),
                    tuple(results.get("notes", {}).get("text", [])),
                )
                _validate_judgement(judgement)
                judgements.append(judgement)

    return judgements


def consolidate(
    judgements: Sequence[Judgement],
    adjudications: Sequence[Judgement] = (),
) -> GoldSample:
    """
    Consolidate independent readings and apply explicit manual adjudications.

    Args:
        judgements: Primary and secondary judgements.
        adjudications: Final decisions after reviewing disagreements.

    Returns:
        Unique gold tasks, unresolved conflicts, and independent agreement counts.

    Raises:
        ValueError: If task contexts differ or adjudications are ambiguous.
    """
    groups: defaultdict[tuple[AlignmentTask, str], list[Judgement]] = defaultdict(list)

    for judgement in judgements:
        groups[judgement.query.task, judgement.query.alignment_id].append(judgement)

    resolved = {
        (item.query.task, item.query.alignment_id): item for item in adjudications
    }

    if len(resolved) != len(adjudications) or not resolved.keys() <= groups.keys():
        raise ValueError("Adjudications must uniquely identify existing tasks")

    gold: list[Judgement] = []
    conflicts: list[str] = []
    repeated = agreed = 0

    for key, readings in groups.items():
        first = readings[0]

        if any(item.query != first.query for item in readings):
            raise ValueError(f"Task context changed: {key}")

        signatures = {(item.status, item.links) for item in readings}
        independent = len({item.annotator for item in readings}) > 1
        repeated += independent
        agreed += independent and len(signatures) == 1
        final = resolved.get(key)

        if final is not None and final.query != first.query:
            raise ValueError(f"Adjudication context changed: {key}")

        if final is None and len(signatures) > 1:
            conflicts.append(first.query.alignment_id)
            continue

        decision = final or first
        notes = tuple(
            dict.fromkeys(note for item in (*readings, decision) for note in item.notes)
        )
        gold.append(
            Judgement(
                first.query, decision.annotator, decision.status, decision.links, notes
            )
        )

    return GoldSample(tuple(gold), tuple(conflicts), repeated, agreed)
