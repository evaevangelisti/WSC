"""
TOML profiles affect inference hypotheses and cache compatibility together.
"""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from wsc.alignment import score_query
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import INSTRUCTIONS_PATH
from wsc.models import POS
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentTask,
    Comparison,
    Definition,
)
from wsc.reading import read_instructions


@pytest.mark.parametrize(
    "profile_name",
    ["baseline", "entailment", "contrastive", "few_shot"],
)
def test_profile_text_changes_model_pairs_and_cache_identity(
    tmp_path: Path,
    profile_name: str,
) -> None:
    """Editing a profile's text invalidates its cached evidence."""
    profiles = read_instructions(INSTRUCTIONS_PATH)
    original = next(profile for profile in profiles if profile.name == profile_name)
    changed = replace(
        original,
        relations={**original.relations, "translation": "Changed relation hypothesis"},
    )
    source = tmp_path / "sample.json"
    _ = source.write_text("{}", encoding="utf-8")
    first = build_metadata(
        source, "model", "revision", "last", 128, instructions=original
    )
    second = build_metadata(
        source, "model", "revision", "last", 128, instructions=changed
    )
    received: list[Comparison] = []

    class Model:
        """Record the public scoring boundary without loading model weights."""

        def score(self, pairs: Sequence[Comparison]) -> list[float]:
            """
            Retain hypotheses for inspection.

            Args:
                pairs: Semantic comparison inputs.

            Returns:
                One positive score for each pair.
            """
            received.extend(pairs)

            return [1.0] * len(pairs)

    query = AlignmentQuery(
        AlignmentTask.TRANSLATIONS,
        "entry",
        "entry",
        "word",
        POS.NOUN,
        (Definition("s", ("sense",)),),
        (Definition("t", ("abbreviation",)),),
    )
    _ = score_query(query, Model(), instructions=changed)

    assert len(profiles) == 4
    assert set(original.relations) == {
        "translation",
        "equivalent",
        "wiktionary_narrower",
        "wiktionary_broader",
    }
    assert cache_key(first) != cache_key(second)
    assert "Changed relation hypothesis" in received[0].query
    assert "Changed relation hypothesis" in second["instruction_text"]
