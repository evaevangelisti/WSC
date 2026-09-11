"""Exercise decision persistence and cache provenance."""

import json
from dataclasses import replace
from pathlib import Path

from wsc.alignment import parse_response, serialize_alignment
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import ALIGNMENT_FIELDS
from wsc.export import TSVWriter
from wsc.models.alignment import GlossMode, ModelSettings
from wsc.reading import read_alignments, read_metadata

from .test_aligner import decision, query


def test_cache_round_trip_preserves_decisions_response_and_synonyms(
    tmp_path: Path,
) -> None:
    """TSV replay retains decisions, tabs, Unicode, and synonyms."""
    sample = replace(query(), lemma='café\t"quoted"\nentry')
    response = json.dumps(
        {"s1": decision("t1"), "s2": None},
        ensure_ascii=False,
        indent=2,
    )
    result = parse_response(sample, response)
    source = tmp_path / "sample.json"
    _ = source.write_text("{}", encoding="utf-8")
    settings = ModelSettings("model")
    metadata = build_metadata(source, settings, GlossMode.FULL)
    path = tmp_path / "translations.tsv"
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        for row in serialize_alignment(result):
            writer.write(row)

    assert tuple(read_alignments(path, {sample.alignment_id: sample})) == (
        replace(result, response=""),
    )
    assert read_metadata(path) == metadata

    for changed in (
        replace(settings, model="other"),
        replace(settings, temperature=0.5),
        replace(settings, maximum_tokens=100),
        replace(
            settings,
            engine_options=(("dtype", "float16"),),
        ),
    ):
        assert cache_key(build_metadata(source, changed, GlossMode.FULL)) != cache_key(
            metadata
        )


def test_reranker_caches_require_explicit_migration(
    tmp_path: Path,
) -> None:
    """Old score tables cannot be interpreted as generated decisions."""
    path = tmp_path / "old.tsv"
    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema": "3"}),
        encoding="utf-8",
    )

    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write(
            {
                "alignment_id": "old",
                "source_id": "source",
                "target_id": "",
                "relation": "",
                "reason": "",
            }
        )

    assert read_metadata(path)["schema"] == "3"
