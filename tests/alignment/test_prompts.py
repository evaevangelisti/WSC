"""Exercise custom task prompts and their provenance."""

from dataclasses import replace
from pathlib import Path

import pytest

from wsc.alignment import build_request, provenance, requests
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import PROMPTS_PATH
from wsc.models.alignment import GlossMode, ModelSettings
from wsc.reading import read_prompts

from .examples import build_query


def test_fingerprints_prompt_edits(
    tmp_path: Path,
) -> None:
    """Prompt content participates in both generation and cache compatibility."""
    original_prompts = read_prompts(PROMPTS_PATH)
    changed_prompts = replace(
        original_prompts,
        tasks={
            **original_prompts.tasks,
            "translations": "Custom prompt: $source_definitions",
        },
    )
    input_path = tmp_path / "sample.json"
    _ = input_path.write_text("{}", encoding="utf-8")
    settings = ModelSettings("model")
    original_metadata = build_metadata(
        input_path, settings, GlossMode.LAST, prompts=original_prompts
    )
    changed_metadata = build_metadata(
        input_path, settings, GlossMode.LAST, prompts=changed_prompts
    )

    assert set(original_prompts.tasks) == {"translations", "wordnet"}
    assert original_prompts.system.startswith("You are a computational lexicographer")
    assert build_request(build_query(), prompts=changed_prompts).prompt.startswith(
        "Custom prompt:",
    )
    assert cache_key(original_metadata) != cache_key(changed_metadata)

    recorded_prompts = changed_metadata["prompts"]

    assert isinstance(recorded_prompts, dict)
    assert "Custom prompt" in recorded_prompts["text"]


@pytest.mark.parametrize("mode", list(GlossMode))
def test_hierarchy_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: GlossMode,
) -> None:
    """Hierarchy edits invalidate only caches whose prompts include the constraint."""
    input_path = tmp_path / "sample.json"
    _ = input_path.write_text("{}", encoding="utf-8")

    settings = ModelSettings("model")
    query = build_query()

    original_request = build_request(query, mode)
    original_metadata = build_metadata(input_path, settings, mode)

    constraint = "- Interpret each hierarchy using the revised instructions."
    monkeypatch.setattr(requests, "HIERARCHY_CONSTRAINT", constraint)
    monkeypatch.setattr(provenance, "HIERARCHY_CONSTRAINT", constraint)

    changed_request = build_request(query, mode)
    changed_metadata = build_metadata(input_path, settings, mode)

    assert (original_request != changed_request) == (mode == GlossMode.FULL)
    assert (cache_key(original_metadata) != cache_key(changed_metadata)) == (
        mode == GlossMode.FULL
    )
