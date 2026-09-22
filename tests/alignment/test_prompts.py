"""Exercise custom alignment task prompts."""

import json
from dataclasses import replace

import pytest

from wsc.alignment import build_request, requests
from wsc.constants import PROMPTS_PATH
from wsc.models.alignment import GlossMode
from wsc.reading import read_prompts

from .examples import build_query


def test_applies_prompt_edits() -> None:
    """Custom prompt content reaches the rendered model request."""
    original = read_prompts(PROMPTS_PATH)
    changed = replace(
        original,
        tasks={
            **original.tasks,
            "translations": "Custom prompt: $source_definitions",
        },
    )

    assert set(original.tasks) == {"translations", "wordnet"}
    assert original.system.startswith("You are a computational lexicographer")
    assert build_request(build_query(), prompts=changed).prompt.startswith(
        "Custom prompt:",
    )


def test_renders_translation_context_as_json_lines() -> None:
    """Translation prompts distinguish source context from target headings."""
    prompt = build_request(build_query()).prompt

    expected_source = json.dumps(
        {
            "id": "s1",
            "tags": ["figurative"],
            "topics": ["finance"],
            "synonyms": ["synonym"],
            "gloss": "first sense",
        },
        separators=(",", ":"),
    )

    assert expected_source in prompt
    assert '{"id":"t1","gloss":"first heading"}' in prompt
    assert "target synonym" not in prompt


@pytest.mark.parametrize("mode", list(GlossMode))
def test_applies_hierarchy_constraint(
    monkeypatch: pytest.MonkeyPatch,
    mode: GlossMode,
) -> None:
    """Hierarchy instructions affect only prompts carrying complete gloss paths."""
    query = build_query()
    original = build_request(query, mode)

    constraint = "- Interpret each hierarchy using the revised instructions."
    monkeypatch.setattr(requests, "HIERARCHY_CONSTRAINT", constraint)

    changed = build_request(query, mode)

    assert (original != changed) == (mode == GlossMode.FULL)
