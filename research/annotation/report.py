"""
Reporting on one set of word offset annotations.

A reading names the export it judged, so several passes may sit in one
project and still be scored apart.
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict, cast

HERE = Path(__file__).resolve().parent

MARKINGS = ("all correct", "some wrong", "nothing marked")
"""What a reading may find of the spans marked, in the order it is offered."""

COVERAGES = ("nothing missing", "something missing")
"""What a reading may find left unmarked, in the order it is offered."""


class Reading(TypedDict):
    """
    One judgement of how an export marked one sentence.
    """

    item_id: str
    source: str
    marking: str
    coverage: str


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

    Precision is asked of the sentences it marked at all, and recall of every
    sentence, a miss counting whether or not anything was marked.

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
        "Precision": (all_correct, len(marked)),
        "Recall": (nothing_missing, len(readings)),
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

    repeated = [judged for judged in judgements.values() if len(judged) > 1]

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
    scored = {name: score(grouped[name]) for name in names}
    rule = f"|{'|'.join(' --- ' for _ in range(len(names) + 1))}|"

    lines = ["# Word offsets", "", f"| Figure | {' | '.join(names)} |", rule]

    for figure in ("Precision", "Recall", "Both", "Marked at all"):
        row = " | ".join(share(*scored[name][figure]) for name in names)
        lines.append(f"| {figure} | {row} |")

    lines += ["", "## What was found", "", f"| Label | {' | '.join(names)} |", rule]

    for label in (*MARKINGS, *COVERAGES):
        field = "marking" if label in MARKINGS else "coverage"
        row = " | ".join(
            str(Counter(reading[field] for reading in grouped[name])[label])
            for name in names
        )

        lines.append(f"| {label} | {row} |")

    lines += ["", "## Agreement with oneself", "", "| Export | Judged the same |"]
    lines.append("| --- | --- |")

    for name in names:
        lines.append(f"| {name} | {share(*agreement(grouped[name]))} |")

    return "\n".join((*lines, ""))


def read_arguments() -> Path:
    """
    Read what the command line settles.

    Returns:
        Which annotation file to report on.
    """
    parser = argparse.ArgumentParser(description="Report on one annotation set.")

    _ = parser.add_argument(
        "annotations",
        type=Path,
        help="the Label Studio export to read",
    )

    return cast(Path, parser.parse_args().annotations)


def main() -> None:
    """
    Score every export the set judged, and write the report beside it.
    """
    annotations = read_arguments()

    readings: list[Reading] = read_json(annotations)

    document = to_markdown(group(readings))

    path = annotations.with_suffix(".md")
    _ = path.write_text(document, encoding="utf-8")

    print(f"Wrote {path}")  # noqa: T201


if __name__ == "__main__":
    main()
