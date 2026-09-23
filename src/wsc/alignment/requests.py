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


def render_source_definition(
    definition: Definition,
    task: AlignmentTask,
    mode: GlossMode,
) -> str:
    """
    Render one Wiktionary sense as a JSON line.

    Args:
        definition: Lexical definition and ancestor context.
        task: Alignment task determining the source fields.
        mode: Wiktionary gloss representation.

    Returns:
        A single JSON object line.
    """
    gloss = (
        " > ".join(definition.glosses)
        if mode == GlossMode.FULL
        else definition.glosses[-1]
    )

    record: dict[str, object] = {"wiktionary_id": definition.id}

    if definition.tags:
        record["tags"] = definition.tags

    if definition.topics:
        record["topics"] = definition.topics

    if definition.synonyms:
        record["synonyms"] = definition.synonyms

    record["gloss"] = gloss

    if task == AlignmentTask.SYNSETS and definition.examples:
        record["examples"] = definition.examples

    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def render_target_definition(
    definition: Definition,
    task: AlignmentTask,
) -> str:
    """
    Render one alignment target as a JSON line.

    Args:
        definition: Candidate definition and optional lexical context.
        task: Alignment task determining the target fields.

    Returns:
        A single JSON object line.
    """
    record: dict[str, object] = {"target_id": definition.id}

    if task == AlignmentTask.SYNSETS:
        if definition.synonyms:
            record["synonyms"] = definition.synonyms

        record["glosses"] = definition.glosses

        if definition.examples:
            record["examples"] = definition.examples
    else:
        record["gloss"] = definition.glosses[-1]

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

    source_definitions = "\n".join(
        render_source_definition(source, query.task, mode)
        for source in query.source_definitions
    )

    target_definitions = "\n".join(
        render_target_definition(target, query.task)
        for target in query.target_definitions
    )

    prompt = Template(prompts.tasks[query.task]).substitute(
        lemma=query.lemma,
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
