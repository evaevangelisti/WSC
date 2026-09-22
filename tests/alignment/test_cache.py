"""Exercise reusable alignment decision persistence."""

import csv
import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import open_alignment_recorder, parse_response
from wsc.constants import ALIGNMENT_FIELDS
from wsc.models.alignment import AlignmentTask
from wsc.reading import read_alignment_cache, read_alignments

from .examples import build_decision, build_query


@pytest.mark.parametrize(
    "assignments",
    [
        (("s1", "t1"), ("s1", "t2"), ("s2", "")),
        (("s1", "t1"), ("s2", "t1")),
    ],
)
def test_validates_cached_equivalences(
    tmp_path: Path,
    assignments: tuple[tuple[str, str], ...],
) -> None:
    """Cached decisions obey the same equivalence constraints as model responses."""
    query = build_query(AlignmentTask.SYNSETS)
    path = tmp_path / "synsets.tsv"

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerows([ALIGNMENT_FIELDS])
        writer.writerows(
            (
                query.alignment_id,
                source,
                target,
                "equivalent" if target else "",
                "The definitions express the same concept." if target else "",
            )
            for source, target in assignments
        )

    with pytest.raises(ValueError, match="One-to-one alignment violated"):
        _ = tuple(read_alignments(path, {query.alignment_id: query}))


@given(
    reason=st.text(min_size=1).filter(lambda text: bool(text.strip())),
    task=st.sampled_from(AlignmentTask),
    abstain=st.booleans(),
)
def test_flushes_replayable_decisions(
    workspace: Callable[[], Path],
    reason: str,
    task: AlignmentTask,
    *,
    abstain: bool,
) -> None:
    """Incremental TSV rows retain Unicode, quoting, links, and abstentions."""
    directory = workspace()
    sample = build_query(task)
    response = json.dumps(
        {
            "s1": [
                {
                    "target_id": "t1",
                    "relation": "translation"
                    if task == AlignmentTask.TRANSLATIONS
                    else "equivalent",
                    "reason": reason,
                },
            ],
            "s2": None
            if abstain
            else build_decision(
                "t2",
                "translation" if task == AlignmentTask.TRANSLATIONS else "equivalent",
            ),
        },
    )
    result = parse_response(sample, response)
    path = directory / f"{task}.tsv"
    queries = {sample.alignment_id: sample}
    expected = (replace(result, response=""),)

    with ExitStack() as stack:
        record = open_alignment_recorder(stack, {task: path})
        record(result)

        assert not path.exists()
        assert (
            tuple(read_alignments(path.with_suffix(".tsv.part"), queries)) == expected
        )

    assert tuple(read_alignments(path, queries)) == expected
    assert set(read_alignment_cache(path)[sample.alignment_id]) == {"s1", "s2"}
    assert not list(directory.glob("*.part"))


def test_preserves_completed_cache(
    tmp_path: Path,
) -> None:
    """An interrupted rewrite preserves the completed decision table."""
    path = tmp_path / "translations.tsv"
    _ = path.write_text("previous decisions", encoding="utf-8")

    def interrupt() -> None:
        """
        Interrupt recording after writing a complete decision.

        Raises:
            RuntimeError: Always, before the cache is published.
        """
        with ExitStack() as stack:
            recorder = open_alignment_recorder(
                stack,
                {AlignmentTask.TRANSLATIONS: path},
            )
            recorder(parse_response(build_query(), '{"s1": null, "s2": null}'))

            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupt()

    assert path.read_text(encoding="utf-8") == "previous decisions"
    assert not list(tmp_path.glob("*.part"))
