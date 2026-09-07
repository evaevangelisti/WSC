"""Exercise decision persistence and cache provenance."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

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
    """TSV replay retains tabs, Unicode, synonyms, and explicit abstentions."""
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

    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata)})

        for row in serialize_alignment(result):
            writer.write(row)

    assert tuple(read_alignments(path)) == (result,)
    assert read_metadata(path) == metadata

    for changed in (
        replace(settings, model="other"),
        replace(settings, temperature=0.5),
        replace(settings, maximum_tokens=100),
        replace(
            settings,
            url="http://localhost:9000/v1",
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

    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps({"schema": "3"})})

    with pytest.raises(ValueError, match="language model schema"):
        _ = list(read_alignments(path))
