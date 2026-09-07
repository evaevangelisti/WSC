"""
Reporting on one set of word offset annotations.

A reading names the export it judged, so several passes may sit in one project and still
be scored apart.
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from .alignment import Annotation, annotator_id

HERE = Path(__file__).resolve().parent.parent

MARKINGS = ("all correct", "some wrong", "nothing marked")
"""What a reading may find of the spans marked, in the order it is offered."""

COVERAGES = ("nothing missing", "something missing")
"""What a reading may find left unmarked, in the order it is offered."""


class Reading(TypedDict):
    """One judgement of how an export marked one sentence."""

    item_id: str
    source: str
    marking: str
    coverage: str
    annotator: NotRequired[str]
    notes: NotRequired[list[str]]


class OffsetTaskData(TypedDict):
    """Identity of an engine-specific offset judgement."""

    task: str
    item_id: str
    source: str


class OffsetTask(TypedDict):
    """Completed Label Studio offset task."""

    data: OffsetTaskData
    annotations: list[Annotation]


def read_offsets(paths: list[Path]) -> list[Reading]:
    """
    Import primary and independent secondary Label Studio offset annotations.

    Args:
        paths: Completed Label Studio JSON exports.

    Returns:
        Source-aware judgements with optional qualitative notes.

    Raises:
        ValueError: If a task lacks either required judgement.
    """
    readings: list[Reading] = []

    for path in paths:
        tasks: list[OffsetTask] = read_json(path)

        for task in tasks:
            for annotation in task["annotations"]:
                if annotation.get("was_cancelled", False):
                    continue

                results = {
                    item["from_name"]: item["value"] for item in annotation["result"]
                }
                marking = results.get("marking", {}).get("choices", [])
                coverage = results.get("coverage", {}).get("choices", [])

                if len(marking) != 1 or len(coverage) != 1:
                    raise ValueError(
                        f"Incomplete offset judgement: {task['data']['item_id']}"
                    )

                if marking[0] not in MARKINGS or coverage[0] not in COVERAGES:
                    raise ValueError(
                        f"Unknown offset judgement: {task['data']['item_id']}"
                    )

                readings.append(
                    {
                        "item_id": task["data"]["item_id"],
                        "source": task["data"]["source"],
                        "marking": marking[0],
                        "coverage": coverage[0],
                        "annotator": annotator_id(annotation["completed_by"]),
                        "notes": results.get("notes", {}).get("text", []),
                    }
                )

    return readings


def resolve_readings(readings: list[Reading]) -> list[Reading]:
    """
    Count repeated tasks once and exclude unresolved offset disagreements.

    Args:
        readings: All primary and secondary judgements for one engine.

    Returns:
        One reading per agreed sentence.
    """
    grouped: defaultdict[str, list[Reading]] = defaultdict(list)

    for reading in readings:
        grouped[reading["item_id"]].append(reading)

    return [
        items[0]
        for items in grouped.values()
        if len({(item["marking"], item["coverage"]) for item in items}) == 1
    ]


def read_json[Payload](
    path: Path,
) -> Payload:  # pyright: ignore[reportInvalidTypeVarUse]
    """
    Read one JSON file.

    Args:
        path: The file to read.

    Returns:
        What it holds, of the shape the caller declares.
    """
    return cast(Payload, json.loads(path.read_text(encoding="utf-8")))


def score(
    readings: list[Reading],
) -> dict[str, tuple[int, int]]:
    """
    Add up what one export was found to have done.

    Precision is asked of the sentences it marked at all, and recall of every sentence,
    a miss counting whether or not anything was marked.

    Args:
        readings: Every judgement of that export.

    Returns:
        The count and the whole behind each figure.
    """
    marked = [reading for reading in readings if reading["marking"] != "nothing marked"]

    all_correct = sum(reading["marking"] == "all correct" for reading in marked)
    nothing_missing = sum(
        reading["coverage"] == "nothing missing" for reading in readings
    )
    both_right = sum(
        reading["marking"] != "some wrong" and reading["coverage"] == "nothing missing"
        for reading in readings
    )

    return {
        "Fully correct marked sentences": (all_correct, len(marked)),
        "Fully covered sentences": (nothing_missing, len(readings)),
        "Both": (both_right, len(readings)),
        "Marked at all": (len(marked), len(readings)),
    }


def agreement(
    readings: list[Reading],
) -> tuple[int, int]:
    """
    Say how often a task judged twice was judged the same way.

    Args:
        readings: Every judgement of one export.

    Returns:
        How many repeated tasks agree, and how many came round again.
    """
    judgements: defaultdict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    for reading in readings:
        judgements[reading["source"], reading["item_id"]].append(
            (reading["marking"], reading["coverage"])
        )

    annotators: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for reading in readings:
        annotators[reading["source"], reading["item_id"]].add(
            reading.get("annotator", "")
        )

    repeated = [
        judged for key, judged in judgements.items() if len(annotators[key]) > 1
    ]

    return sum(len(set(judged)) == 1 for judged in repeated), len(repeated)


def share(
    part: int,
    whole: int,
) -> str:
    """
    Write what part of a whole a count is.

    Args:
        part: The count.
        whole: What it is part of.

    Returns:
        The share and the counts behind it, or nothing where there is no whole.
    """
    return f"{part / whole:.1%} ({part}/{whole})" if whole else ""


def group(
    readings: Iterable[Reading],
) -> dict[str, list[Reading]]:
    """
    Sort the readings by the export each one judged.

    Args:
        readings: What the annotator wrote.

    Returns:
        The readings of each export, named as the export is.
    """
    grouped: defaultdict[str, list[Reading]] = defaultdict(list)

    for reading in readings:
        grouped[reading["source"]].append(reading)

    return dict(grouped)


def to_markdown(
    grouped: dict[str, list[Reading]],
) -> str:
    """
    Write the report as Markdown, one column per export.

    Args:
        grouped: The readings of each export.

    Returns:
        The document.
    """
    names = sorted(grouped)
    resolved = {name: resolve_readings(grouped[name]) for name in names}
    shared: set[str] = (
        set[str].intersection(
            *({item["item_id"] for item in resolved[name]} for name in names)
        )
        if names
        else set()
    )
    comparable = {
        name: [item for item in resolved[name] if item["item_id"] in shared]
        for name in names
    }
    scored = {name: score(comparable[name]) for name in names}
    rule = f"|{'|'.join(' --- ' for _ in range(len(names) + 1))}|"

    lines = [
        "# Word offsets",
        "",
        f"Shared resolved sentences: {len(shared)}.",
        "",
        "Figures describe whole sentences, not individual-span precision or recall.",
        "",
        f"| Figure | {' | '.join(names)} |",
        rule,
    ]

    for figure in (
        "Fully correct marked sentences",
        "Fully covered sentences",
        "Both",
        "Marked at all",
    ):
        row = " | ".join(share(*scored[name][figure]) for name in names)
        lines.append(f"| {figure} | {row} |")

    lines += ["", "## What was found", "", f"| Label | {' | '.join(names)} |", rule]

    for label in (*MARKINGS, *COVERAGES):
        field = "marking" if label in MARKINGS else "coverage"
        row = " | ".join(
            str(Counter(reading[field] for reading in comparable[name])[label])
            for name in names
        )

        lines.append(f"| {label} | {row} |")

    lines += [
        "",
        "## Independent annotation agreement",
        "",
        "| Export | Judged the same |",
    ]
    lines.append("| --- | --- |")

    for name in names:
        lines.append(f"| {name} | {share(*agreement(grouped[name]))} |")

    lines += [
        "",
        "## Annotator notes",
        "",
        "| Export | Sentence | Notes |",
        "| --- | --- | --- |",
    ]

    for name in names:
        notes = {
            (reading["item_id"], note)
            for reading in grouped[name]
            for note in reading.get("notes", [])
        }

        for identifier, note in sorted(notes):
            text = note.replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {name} | {identifier} | {text} |")

    return "\n".join((*lines, ""))


def read_arguments() -> list[Path]:
    """
    Read what the command line settles.

    Returns:
        Which annotation file to report on.
    """
    parser = argparse.ArgumentParser(description="Report on one annotation set.")

    _ = parser.add_argument(
        "annotations",
        type=Path,
        nargs="+",
        help="the Label Studio export to read",
    )

    return cast(list[Path], parser.parse_args().annotations)


def main() -> None:
    """Score every export the set judged, and write the report beside it."""
    annotations = read_arguments()

    readings = read_offsets(annotations)

    document = to_markdown(group(readings))

    path = annotations[0].with_suffix(".md")
    _ = path.write_text(document, encoding="utf-8")

    print(f"Wrote {path}")  # noqa: T201


if __name__ == "__main__":
    main()
