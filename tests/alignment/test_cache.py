"""Exercise decision persistence and cache provenance."""

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
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import ALIGNMENT_FIELDS
from wsc.models.alignment import AlignmentTask, GlossMode, ModelSettings
from wsc.reading import read_alignments, read_metadata

from .examples import build_decision, build_query


@pytest.mark.parametrize(
    "assignments",
    [
        (("s1", "t1"), ("s1", "t2"), ("s2", "")),
        (("s1", "t1"), ("s2", "t1")),
    ],
)
def test_cached_equivalences_require_unique_sources_and_synsets(
    tmp_path: Path,
    assignments: tuple[tuple[str, str], ...],
) -> None:
    """Cached decisions obey the same equivalence constraints as model responses."""
    query = build_query(AlignmentTask.WORDNET)
    path = tmp_path / "wordnet.tsv"

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
def test_recorded_decisions_are_visible_before_commit_and_replay_exactly(
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
    metadata: dict[str, object] = {"schema": "test"}
    queries = {sample.alignment_id: sample}
    expected = (replace(result, response=""),)

    with ExitStack() as stack:
        record = open_alignment_recorder(stack, {task: path}, metadata)
        record(result)

        assert not path.exists()
        assert (
            tuple(read_alignments(path.with_suffix(".tsv.part"), queries)) == expected
        )
        assert not (directory / "metadata.json").exists()

    assert tuple(read_alignments(path, queries)) == expected
    assert read_metadata(path) == metadata
    assert not list(directory.glob("*.part"))


@pytest.mark.parametrize(
    "settings",
    [
        ModelSettings("other"),
        ModelSettings("model", temperature=0.5),
        ModelSettings("model", maximum_tokens=100),
        ModelSettings("model", engine_options=(("dtype", "float16"),)),
        ModelSettings("model", reasoning_parser="qwen3"),
        ModelSettings("model", reasoning_effort="low"),
        ModelSettings("model", chat_template_options=(("enable_thinking", False),)),
    ],
)
def test_generation_settings_change_cache_identity(
    tmp_path: Path,
    settings: ModelSettings,
) -> None:
    """Every generation option contributes to cache compatibility."""
    source = tmp_path / "sample.json"
    _ = source.write_text("{}", encoding="utf-8")
    original = build_metadata(source, ModelSettings("model"), GlossMode.FULL)
    changed = build_metadata(source, settings, GlossMode.FULL)

    assert cache_key(original) != cache_key(changed)


def test_failed_recording_preserves_completed_cache(
    tmp_path: Path,
) -> None:
    """An interrupted run replaces neither completed decisions nor metadata."""
    path = tmp_path / "translations.tsv"
    metadata_path = tmp_path / "metadata.json"
    _ = path.write_text("previous decisions", encoding="utf-8")
    _ = metadata_path.write_text("previous metadata", encoding="utf-8")

    def interrupt() -> None:
        with ExitStack() as stack:
            recorder = open_alignment_recorder(
                stack,
                {AlignmentTask.TRANSLATIONS: path},
                {"schema": "new"},
            )
            recorder(parse_response(build_query(), '{"s1": null, "s2": null}'))

            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupt()

    assert path.read_text(encoding="utf-8") == "previous decisions"
    assert metadata_path.read_text(encoding="utf-8") == "previous metadata"
    assert not list(tmp_path.glob("*.part"))
