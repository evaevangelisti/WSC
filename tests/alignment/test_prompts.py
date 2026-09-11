"""Exercise custom task prompts and their provenance."""

from dataclasses import replace
from pathlib import Path

from wsc.alignment import build_request
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import PROMPTS_PATH
from wsc.models.alignment import GlossMode, ModelSettings
from wsc.reading import read_prompts

from .test_aligner import query


def test_prompt_edits_change_requests_and_cache_identity(
    tmp_path: Path,
) -> None:
    """Prompt content participates in both generation and cache compatibility."""
    original = read_prompts(PROMPTS_PATH)
    changed = replace(
        original,
        tasks={**original.tasks, "translations": "Custom prompt: $source_definitions"},
    )
    source = tmp_path / "sample.json"
    _ = source.write_text("{}", encoding="utf-8")
    settings = ModelSettings("model")
    first = build_metadata(source, settings, GlossMode.LAST, prompts=original)
    second = build_metadata(source, settings, GlossMode.LAST, prompts=changed)

    assert set(original.tasks) == {"translations", "wordnet"}
    assert original.system.startswith("You are a computational lexicographer")
    assert build_request(query(), prompts=changed).prompt.startswith("Custom prompt:")
    assert cache_key(first) != cache_key(second)
    prompts = second["prompts"]
    assert isinstance(prompts, dict)
    assert "Custom prompt" in prompts["text"]
