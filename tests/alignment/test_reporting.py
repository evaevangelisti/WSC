"""Exercise alignment report publication through its public API."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import parse_response
from wsc.alignment.reporting import AlignmentStatistics, write_alignment
from wsc.models import POS, Lemma
from wsc.models.alignment import AlignmentTask

from .examples import build_decision, build_query


@given(
    assignments=st.lists(st.tuples(st.booleans(), st.booleans()), max_size=20),
    tasks=st.lists(st.sampled_from(AlignmentTask), unique=True),
)
def test_counts_published_decisions(
    workspace: Callable[[], Path],
    assignments: list[tuple[bool, bool]],
    tasks: list[AlignmentTask],
) -> None:
    """Reports account for links and abstentions, including empty task selections."""
    statistics = AlignmentStatistics(tasks)
    output_dir = workspace() / "alignment"

    def entries() -> Iterator[Lemma]:
        """Accumulate resolved decisions while publication consumes each entry."""
        for index, (first, second) in enumerate(assignments):
            for task in tasks:
                relation = (
                    "translation"
                    if task == AlignmentTask.TRANSLATIONS
                    else "equivalent"
                )
                statistics.add(
                    parse_response(
                        build_query(task),
                        json.dumps(
                            {
                                "s1": build_decision("t1" if first else None, relation),
                                "s2": build_decision(
                                    "t2" if second else None, relation
                                ),
                            }
                        ),
                    ),
                )

            yield Lemma(f"word{index}.noun", "word", POS.NOUN)

    write_alignment(entries(), output_dir, statistics, {"source": "collection"})

    manifest = cast(
        dict[str, object], json.loads((output_dir / "manifest.json").read_text())
    )
    totals = cast(dict[str, object], manifest["totals"])
    accepted = sum(first + second for first, second in assignments)
    expected = {
        "evaluated": 2 * len(assignments),
        "aligned": accepted,
        "unaligned": 2 * len(assignments) - accepted,
    }

    assert manifest["source"] == "collection"
    assert manifest["files"] == {
        "senses": "senses.jsonl",
        "reports": {task: f"reports/{task}.json" for task in tasks},
        "manifest": "manifest.json",
    }
    assert len((output_dir / "senses.jsonl").read_text().splitlines()) == len(
        assignments
    )
    assert {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    } == {"senses.jsonl", "manifest.json", *(f"reports/{task}.json" for task in tasks)}

    for task in tasks:
        report = cast(
            dict[str, object],
            json.loads((output_dir / "reports" / f"{task}.json").read_text()),
        )
        relation = "translation" if task == AlignmentTask.TRANSLATIONS else "equivalent"

        assert totals[task] == report["senses"] == expected
        assert report["associations"] == accepted
        assert report["relations"] == ({relation: accepted} if accepted else {})


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["inference", "serialization"])
def test_preserves_unpublished_alignment(
    tmp_path: Path,
    failure: str,
    *,
    existing: bool,
) -> None:
    """An interrupted run preserves previous files and removes staged output."""
    output_dir = tmp_path / "alignment"
    statistics = AlignmentStatistics(AlignmentTask)
    entry = Lemma("word.noun", "word", POS.NOUN)
    expected: dict[Path, bytes] = {}

    if existing:
        write_alignment([entry], output_dir, statistics, {})
        expected = {
            path.relative_to(output_dir): path.read_bytes()
            for path in output_dir.rglob("*")
            if path.is_file()
        }

    def interrupted_entries() -> Iterator[Lemma]:
        """
        Interrupt inference after a staged entry.

        Yields:
            One entry before inference fails.

        Raises:
            RuntimeError: After the first entry is consumed.
        """
        yield entry

        raise RuntimeError("Interrupted inference")

    if failure == "inference":
        with pytest.raises(RuntimeError, match="Interrupted inference"):
            write_alignment(interrupted_entries(), output_dir, statistics, {})
    else:
        with pytest.raises(TypeError, match="not JSON serializable"):
            write_alignment([entry], output_dir, statistics, {"unsupported": object()})

    assert {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    } == expected
    assert output_dir.exists() == existing
    assert not list(tmp_path.glob(".alignment-*"))
