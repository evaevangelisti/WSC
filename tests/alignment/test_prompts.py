"""Exercise custom alignment task prompts."""

import json
from dataclasses import replace

import pytest

from wsc.alignment import build_request, requests
from wsc.constants import PROMPTS_PATH
from wsc.models.alignment import AlignmentTask, Definition, GlossMode
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

    assert set(original.tasks) == {"translations", "synsets"}
    assert original.system.startswith("You are a computational lexicographer")
    assert build_request(build_query(), prompts=changed).prompt.startswith(
        "Custom prompt:",
    )


def test_renders_translation_context_as_json_lines() -> None:
    """Translation prompts distinguish source context from target headings."""
    prompt = build_request(build_query()).prompt

    expected_source = json.dumps(
        {
            "wiktionary_id": "s1",
            "tags": ["figurative"],
            "topics": ["finance"],
            "synonyms": ["synonym"],
            "gloss": "first sense",
        },
        separators=(",", ":"),
    )

    assert expected_source in prompt
    assert '{"target_id":"t1","gloss":"first heading"}' in prompt
    assert "target synonym" not in prompt


def test_omits_source_examples_from_translation_prompts() -> None:
    """Translation prompts omit Wiktionary sentence examples."""
    query = replace(
        build_query(),
        source_definitions=(
            Definition(
                "s1",
                ("first sense",),
                examples=("A Wiktionary sentence example.",),
            ),
        ),
    )

    prompt = build_request(query).prompt

    assert "A Wiktionary sentence example." not in prompt


def test_renders_distinct_synset_glosses() -> None:
    """Synset glosses remain separate input strings for the model."""
    query = replace(
        build_query(AlignmentTask.SYNSETS),
        target_definitions=(
            Definition(
                "synset-1",
                ("First source gloss.", "Second source gloss."),
                synonyms=("term",),
                examples=("An example.",),
            ),
        ),
    )

    prompt = build_request(query).prompt

    assert (
        '"target_id":"synset-1","synonyms":["term"],'
        '"glosses":["First source gloss.","Second source gloss."],'
        '"examples":["An example."]'
    ) in prompt
    assert "First source gloss. > Second source gloss." not in prompt


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
