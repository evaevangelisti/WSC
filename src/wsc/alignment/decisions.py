"""Decode and validate lexical alignment decisions."""

import json
from typing import cast

from ..errors import InvalidModelResponseError
from ..models import SynsetRelation
from ..models.alignment import (
    AlignmentDecision,
    AlignmentLink,
    AlignmentQuery,
    AlignmentResult,
)
from .tasks import TASK_HANDLERS


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

        exclusive_targets = [
            link.target_id
            for link in decision.links
            if handler.one_to_one or link.relation == SynsetRelation.EQUIVALENT
        ]

        if len(exclusive_targets) > 1 or assigned_targets.intersection(
            exclusive_targets,
        ):
            raise ValueError(
                f"One-to-one alignment violated\nQuery: {query.alignment_id}\n"
                + f"Source: {decision.source_id}"
            )

        assigned_targets.update(exclusive_targets)

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
            f"Invalid JSON response for {query.alignment_id}\n"
            + f"{error.msg} at line {error.lineno}, column {error.colno}\n\n"
            + f"Response\n{response}"
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
