"""Render model requests and validate generated lexical associations."""

import json
from string import Template
from typing import cast

from ..constants import DEFAULT_PROMPTS, HIERARCHY_CONSTRAINT
from ..errors import InvalidModelResponseError
from ..models.alignment import (
    AlignmentDecision,
    AlignmentLink,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    Definition,
    GlossMode,
    LanguageModel,
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

    prompt = Template(prompts.tasks[query.task]).substitute(
        lemma=json.dumps(query.lemma, ensure_ascii=False),
        pos=query.pos,
        source_definitions="\n".join(
            render_definition(source, mode) for source in query.source_definitions
        ),
        target_definitions="\n".join(
            render_definition(target, GlossMode.LAST)
            for target in query.target_definitions
        ),
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


def validate_result(
    result: AlignmentResult,
) -> None:
    """
    Validate decision identities, relations, reasons, and task cardinality.

    Args:
        result: Generated or persisted decisions.

    Raises:
        ValueError: If decisions violate the task contract.
    """
    query = result.query

    sources = [source.id for source in query.source_definitions]
    decisions = [decision.source_id for decision in result.decisions]

    if len(set(sources)) != len(sources) or sorted(sources) != sorted(decisions):
        raise ValueError(f"Expected one decision per source: {query.alignment_id}")

    handler = TASK_HANDLERS[query.task]
    targets = {target.id for target in query.target_definitions}

    assigned_targets: set[str] = set()

    for decision in result.decisions:
        linked_targets = [link.target_id for link in decision.links]

        if len(set(linked_targets)) != len(linked_targets):
            raise ValueError(f"Repeated target association: {decision.source_id}")

        if handler.one_to_one and (
            len(linked_targets) > 1 or assigned_targets.intersection(linked_targets)
        ):
            raise ValueError(f"One-to-one alignment violated: {query.alignment_id}")

        assigned_targets.update(linked_targets)

        for link in decision.links:
            if (
                link.source_id != decision.source_id
                or link.target_id not in targets
                or link.relation not in handler.relations
                or not link.reason.strip()
            ):
                raise ValueError(f"Invalid association: {decision.source_id}")
def _parse_link(
    source_id: str,
    value: object,
) -> AlignmentLink:
    """
    Decode one association from untrusted model output.

    Args:
        source_id: Source identifier enclosing the association.
        value: Decoded JSON value.

    Returns:
        A typed association with its supporting reason.

    Raises:
        ValueError: If the association fields are invalid.
    """
    if not isinstance(value, dict):
        raise ValueError(f"Expected association object: {source_id}")

    fields = cast(dict[str, object], value)

    if set(fields) != {"target_id", "relation", "reason"} or not all(
        isinstance(item, str) for item in fields.values()
    ):
        raise ValueError(f"Invalid association fields: {source_id}")

    return AlignmentLink(
        source_id,
        cast(str, fields["target_id"]),
        cast(str, fields["relation"]),
        cast(str, fields["reason"]),
    )


def parse_response(
    query: AlignmentQuery,
    response: str,
) -> AlignmentResult:
    """
    Decode JSON associations and explicit null decisions.

    Args:
        query: Candidate identities supplied to the model.
        response: Generated JSON object.

    Returns:
        Validated associations and the original response.

    Raises:
        ValueError: If the response contains invalid JSON or associations.
    """
    try:
        decoded = cast(object, json.loads(response))
    except json.JSONDecodeError as error:
        raise InvalidModelResponseError(
            f"Invalid JSON response for {query.alignment_id}: {error}"
        ) from error

    if not isinstance(decoded, dict):
        raise ValueError(f"Expected a JSON object: {query.alignment_id}")

    records = cast(dict[str, object], decoded)

    if set(records) != {source.id for source in query.source_definitions}:
        raise ValueError(f"Expected all source identifiers: {query.alignment_id}")

    decisions: list[AlignmentDecision] = []

    for source in query.source_definitions:
        value = records[source.id]
        links: tuple[AlignmentLink, ...] = ()

        if value is not None:
            if not isinstance(value, list) or not value:
                raise ValueError(f"Expected associations or null: {source.id}")

            links = tuple(
                _parse_link(source.id, item) for item in cast(list[object], value)
            )

        decisions.append(AlignmentDecision(source.id, links))

    result = AlignmentResult(query, tuple(decisions), response)
    validate_result(result)

    return result


def align_query(
    query: AlignmentQuery,
    model: LanguageModel,
    mode: GlossMode = GlossMode.LAST,
    prompts: AlignmentPrompts = DEFAULT_PROMPTS,
) -> AlignmentResult:
    """
    Generate validated decisions for a complete alignment query.

    Args:
        query: Sources and available candidates.
        model: Language model generation boundary.
        mode: Wiktionary gloss representation.
        prompts: Task prompt templates.

    Returns:
        Associations or explicit abstentions for every source.
    """
    if not query.target_definitions or not query.source_definitions:
        return AlignmentResult(
            query,
            tuple(AlignmentDecision(source.id) for source in query.source_definitions),
        )

    request = build_request(query, mode, prompts)

    try:
        response = model.generate(request)
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}: {error}"
        ) from error

    try:
        return parse_response(query, response)
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Invalid model response for {query.alignment_id}: {error}"
        ) from error
