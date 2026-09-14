"""Shared sampling and manual alignment reference contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
from annotation.scripts.alignment import (
    AnnotatedTask,
    Judgement,
    consolidate,
    read_judgements,
    write_tasks,
)
from annotation.scripts.offsets import build_offsets
from annotation.scripts.report import read_json
from annotation.scripts.sampling import EXPORTS, sample_items, split_queries

from wsc.models import POS
from wsc.models.alignment import AlignmentQuery, AlignmentTask, Definition
from wsc.reading import QueryRecord


def query(
    identifier: str = "entry",
) -> AlignmentQuery:
    """
    Build a complete translation annotation input.

    Args:
        identifier: Stable entry identifier and headword.

    Returns:
        Two senses and two translation groups.
    """
    return AlignmentQuery(
        AlignmentTask.TRANSLATIONS,
        identifier,
        identifier,
        identifier,
        POS.NOUN,
        (Definition("s1", ("first",)), Definition("s2", ("second",))),
        (
            Definition("t1", ("abbreviation one",)),
            Definition("t2", ("abbreviation two",)),
        ),
    )


def annotated_task(
    sample: AlignmentQuery,
    links: list[tuple[str, str, str]],
    status: str,
    annotator: int = 1,
    notes: str = "",
) -> AnnotatedTask:
    """
    Build a standard Label Studio export with completed controls.

    Args:
        sample: Original model-blind query.
        links: Manually selected associations.
        status: Overall annotation decision.
        annotator: Independent annotator identifier.
        notes: Optional free text.

    Returns:
        One completed Label Studio task.
    """
    return {
        "data": {
            "task": sample.task,
            "alignment_id": sample.alignment_id,
            "query": cast(QueryRecord, json.loads(json.dumps(asdict(sample)))),
        },
        "annotations": [
            {
                "completed_by": annotator,
                "result": [
                    {"from_name": "status", "value": {"choices": [status]}},
                    {
                        "from_name": "links",
                        "value": {"choices": [json.dumps(link) for link in links]},
                    },
                    {"from_name": "notes", "value": {"text": [notes] if notes else []}},
                ],
            },
        ],
    }


def test_preserves_engine_proposals(
    tmp_path: Path,
) -> None:
    """All engines receive identical sentences and independent repeat identities."""
    input_directory = tmp_path / "data"
    input_directory.mkdir()

    for position, export in enumerate(EXPORTS):
        records = [
            {
                "id": f"word{index}.noun",
                "lemma": f"word{index}",
                "pos": "noun",
                "senses": [
                    {
                        "id": f"sense{index}",
                        "glosses": ["sense"],
                        "sentences": [
                            {
                                "text": "a <word> example",
                                "word_offsets": [
                                    {"offset": [2, 8], "sources": ["bold"]},
                                    {"offset": [3, 7], "sources": ["lemmatizer"]},
                                ]
                                if position
                                else [],
                            },
                        ],
                    },
                ],
            }
            for index in range(130)
        ]

        if position == 2:
            records.reverse()

        _ = (input_directory / export).write_text(
            "\n".join(json.dumps(record) for record in records),
            encoding="utf-8",
        )

    output_directory = tmp_path / "tasks"
    build_offsets(input_directory, output_directory, 100, 31)
    tasks: list[list[dict[str, dict[str, object]]]] = [
        read_json(output_directory / f"{Path(export).stem}.json") for export in EXPORTS
    ]
    repeats: list[list[dict[str, dict[str, object]]]] = [
        read_json(output_directory / f"{Path(export).stem}.second.json")
        for export in EXPORTS
    ]

    assert all(len(items) == 100 for items in tasks)
    assert all(len(items) == 10 for items in repeats)
    assert (
        len({tuple(item["data"]["item_id"] for item in items) for items in tasks}) == 1
    )
    assert (
        len({tuple(item["data"]["item_id"] for item in items) for items in repeats})
        == 1
    )
    assert tasks[0][0]["data"]["marked"] == "a &lt;word&gt; example"
    assert tasks[1][0]["data"]["marked"] == "a <b>&lt;word&gt;</b> example"


def test_excludes_model_predictions(
    tmp_path: Path,
) -> None:
    """One shared sample supplies one hundred primary and ten secondary tasks."""
    queries = [query(f"entry{index}") for index in range(100)]
    write_tasks(queries, tmp_path, 4)
    tasks: list[dict[str, dict[str, object]]] = read_json(tmp_path / "tasks.json")
    repeats: list[dict[str, dict[str, object]]] = read_json(
        tmp_path / "tasks.second.json",
    )
    sample: dict[str, dict[str, str]] = read_json(tmp_path / "sample.json")

    assert len(tasks) == 100
    assert len(repeats) == 10
    assert all("predictions" not in task for task in tasks)
    assert all("model" not in task["data"] for task in tasks)
    assert set(sample["splits"].values()) == {"development", "test"}


def test_consolidates_repeated_annotations(
    tmp_path: Path,
) -> None:
    """Repeated readings contribute to agreement statistics."""
    path = tmp_path / "labels.json"
    sample = query()
    _ = path.write_text(
        json.dumps(
            [
                annotated_task(
                    sample,
                    [("s1", "t1", "translation")],
                    "matched",
                    1,
                    "first note",
                ),
                annotated_task(
                    sample,
                    [("s1", "t1", "translation")],
                    "matched",
                    2,
                    "second note",
                ),
            ],
        ),
        encoding="utf-8",
    )
    gold = consolidate(read_judgements([path]))

    assert len(gold.judgements) == 1
    assert gold.repeated == gold.agreed == 1
    assert gold.judgements[0].notes == ("first note", "second note")


def test_requires_explicit_adjudication() -> None:
    """Disagreement is neither a negative label nor a majority vote."""
    first = Judgement(
        query(),
        "1",
        "matched",
        frozenset({("s1", "t1", "translation")}),
        (),
    )
    second = Judgement(query(), "2", "no_match", frozenset(), ())
    unresolved = consolidate([first, second])
    resolved = consolidate([first, second], [first])

    assert unresolved.conflicts == ("entry",)
    assert not unresolved.judgements
    assert resolved.judgements == (first,)
    assert resolved.repeated == 1
    assert resolved.agreed == 0


@pytest.mark.parametrize(
    ("links", "status"),
    [
        ([("s1", "t1", "translation"), ("s1", "t2", "translation")], "matched"),
        ([("s1", "t1", "translation")], "no_match"),
        ([], "matched"),
        ([("unknown", "t1", "translation")], "matched"),
    ],
)
def test_rejects_invalid_annotations(
    tmp_path: Path,
    links: list[tuple[str, str, str]],
    status: str,
) -> None:
    """Contradictory manual judgements cannot enter gold."""
    path = tmp_path / "labels.json"
    _ = path.write_text(
        json.dumps([annotated_task(query(), links, status)]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="entry"):
        _ = read_judgements([path])


def test_reproduces_grouped_sampling() -> None:
    """Sampling is deterministic and headword groups cannot leak across splits."""
    assert sample_items(range(1000), 100, 41) == sample_items(range(1000), 100, 41)

    queries = [query(f"entry{index}") for index in range(20)]
    extra = AlignmentQuery(
        AlignmentTask.WORDNET,
        "other",
        "entry1.verb",
        "entry1",
        POS.VERB,
        (Definition("s", ("verb",)),),
        (),
    )
    splits = split_queries([*queries, extra], 41)

    assert splits["other"] == splits["entry1"]


@pytest.mark.parametrize("task", list(AlignmentTask))
def test_rejects_removed_status(
    tmp_path: Path,
    task: AlignmentTask,
) -> None:
    """Removed decisions require manual revision."""
    sample = replace(query(), task=task)
    path = tmp_path / "labels.json"
    _ = path.write_text(
        json.dumps([annotated_task(sample, [], "candidate_missing")]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown judgement"):
        _ = read_judgements([path])
