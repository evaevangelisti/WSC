"""Render lexical definitions, task prompts, and constrained response schemas."""

import json
from string import Template

from ..constants import DEFAULT_PROMPTS, HIERARCHY_CONSTRAINT
from ..models.alignment import (
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentTask,
    Definition,
    GlossMode,
    ModelRequest,
)
from .tasks import TASK_HANDLERS


def render_definition(
    definition: Definition,
    mode: GlossMode,
) -> str:
    """
    Render an identifier, optional synonyms, and a definition.

    Args:
        definition: Lexical definition and ancestor context.
        mode: Source gloss representation.

    Returns:
        A definition line suitable for a model prompt.
    """
    synonyms = f" ({', '.join(definition.synonyms)})" if definition.synonyms else ""
    gloss = (
        " > ".join(definition.glosses)
        if mode == GlossMode.FULL
        else definition.glosses[-1]
    )

    return f"{definition.id}{synonyms} {gloss}"


def _render_translation_definition(
    definition: Definition,
    mode: GlossMode,
    *,
    source: bool,
) -> str:
    """
    Render one translation definition as an unambiguous JSON line.

    Args:
        definition: Lexical definition and optional source context.
        mode: Source gloss representation.
        source: Whether to include Wiktionary metadata.

    Returns:
        A single JSON object line.
    """
    gloss = (
        " > ".join(definition.glosses)
        if mode == GlossMode.FULL
        else definition.glosses[-1]
    )

    record: dict[str, object] = {"id": definition.id}

    if source:
        if definition.tags:
            record["tags"] = definition.tags

        if definition.topics:
            record["topics"] = definition.topics

        if definition.synonyms:
            record["synonyms"] = definition.synonyms

    record["gloss"] = gloss

    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def build_request(
    query: AlignmentQuery,
    mode: GlossMode = GlossMode.LAST,
    prompts: AlignmentPrompts = DEFAULT_PROMPTS,
) -> ModelRequest:
    """
    Build a task prompt and its constrained JSON schema.

    Args:
        query: Complete source and candidate definitions.
        mode: Wiktionary gloss representation.
        prompts: Task prompt templates.

    Returns:
        A prompt and schema requiring one decision per source.
    """
    handler = TASK_HANDLERS[query.task]

    if query.task == AlignmentTask.TRANSLATIONS:
        source_definitions = "\n".join(
            _render_translation_definition(source, mode, source=True)
            for source in query.source_definitions
        )

        target_definitions = "\n".join(
            _render_translation_definition(target, GlossMode.LAST, source=False)
            for target in query.target_definitions
        )
    else:
        source_definitions = "\n".join(
            render_definition(source, mode) for source in query.source_definitions
        )

        target_definitions = "\n".join(
            render_definition(target, GlossMode.LAST)
            for target in query.target_definitions
        )

    prompt = Template(prompts.tasks[query.task]).substitute(
        lemma=json.dumps(query.lemma, ensure_ascii=False),
        pos=query.pos,
        source_definitions=source_definitions,
        target_definitions=target_definitions,
        hierarchy=HIERARCHY_CONSTRAINT if mode == GlossMode.FULL else "",
    )

    link_schema = {
        "type": "object",
        "properties": {
            "target_id": {
                "type": "string",
                "enum": [target.id for target in query.target_definitions],
            },
            "relation": {"type": "string", "enum": list(handler.relations)},
            "reason": {"type": "string"},
        },
        "required": ["target_id", "relation", "reason"],
        "additionalProperties": False,
    }

    associations: dict[str, object] = {
        "type": "array",
        "items": link_schema,
        "minItems": 1,
    }

    if handler.one_to_one:
        associations["maxItems"] = 1

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            source.id: {"anyOf": [associations, {"type": "null"}]}
            for source in query.source_definitions
        },
        "required": [source.id for source in query.source_definitions],
        "additionalProperties": False,
    }

    return ModelRequest(prompts.system, prompt, schema)
