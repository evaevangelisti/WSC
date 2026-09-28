"""
Generate and validate decisions for one alignment query.
"""

from ...constants import DEFAULT_PROMPTS
from ...errors import InvalidModelResponseError
from ...models.alignment import (
    AlignmentDecision,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    GlossMode,
    LanguageModel,
    ModelOutcome,
)
from ..decisions import parse_response
from ..requests import build_request


def _abstain(
    query: AlignmentQuery,
) -> AlignmentResult:
    """
    Build empty decisions when a query has nothing to compare.

    Args:
        query: Query without sources or candidates.

    Returns:
        One empty decision per source definition.
    """
    return AlignmentResult(
        query,
        tuple(AlignmentDecision(source.id) for source in query.source_definitions),
    )


def parse_outcome(
    query: AlignmentQuery,
    outcome: ModelOutcome,
) -> AlignmentResult:
    """
    Validate one generated outcome against its query.

    Args:
        query: Sources and candidates supplied to the model.
        outcome: Generated text or its failure description.

    Returns:
        Validated alignment decisions.

    Raises:
        InvalidModelResponseError: If generation or validation failed.
    """
    if outcome.text is None:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}\n\n{outcome.error}"
        )

    try:
        return parse_response(query, outcome.text)
    except InvalidModelResponseError:
        raise
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Invalid model response for {query.alignment_id}\n"
            + f"{error}\n\nResponse\n{outcome.text}"
        ) from error


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
        mode: Wiktionary definition representation.
        prompts: Task prompt templates.

    Returns:
        Associations or explicit abstentions for every source.

    Raises:
        InvalidModelResponseError: If generation fails or the model response is invalid.
    """
    if not query.target_definitions or not query.source_definitions:
        return _abstain(query)

    request = build_request(query, mode, prompts)

    try:
        response = model.generate(request)
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}\n\n{error}"
        ) from error

    return parse_outcome(query, ModelOutcome(response))
