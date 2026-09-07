"""
Semantic scoring and constrained association selection.
"""

from math import isfinite

from ..constants import DEFAULT_INSTRUCTIONS, TRANSLATION_RELATION
from ..models import WordNetRelation
from ..models.alignment import (
    AlignmentInstructions,
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Comparison,
    Definition,
    GlossMode,
    Scorer,
)


def render_definition(
    definition: Definition,
    mode: GlossMode,
) -> str:
    """
    Render the configured gloss representation.

    Args:
        definition: Candidate and its ancestor definitions.
        mode: Representation under evaluation.

    Returns:
        Text presented to the semantic model.
    """
    gloss = definition.glosses[-1]

    if mode == GlossMode.FULL:
        return " ".join(definition.glosses)

    if mode == GlossMode.CONTEXT and len(definition.glosses) > 1:
        return (
            f"Definition: {gloss}\n"
            f"Ancestor context: {' > '.join(definition.glosses[:-1])}"
        )

    return gloss


def score_query(
    query: AlignmentQuery,
    scorer: Scorer,
    mode: GlossMode = GlossMode.LAST,
    instructions: AlignmentInstructions = DEFAULT_INSTRUCTIONS,
) -> AlignmentResult:
    """
    Evaluate every candidate relation without lexical shortcuts.

    Args:
        query: Complete source and target sets.
        scorer: Cross-encoder scoring boundary.
        mode: Source gloss representation.
        instructions: Relation hypotheses paired with the model's general instruction.

    Returns:
        Scores retained before matching or abstention.

    Raises:
        ValueError: If the model returns missing or nonfinite scores.
    """
    relations = (
        (TRANSLATION_RELATION,)
        if query.task == AlignmentTask.TRANSLATIONS
        else tuple(WordNetRelation)
    )

    candidates = [
        (source_definition, target_definition, relation)
        for source_definition in query.source_definitions
        for target_definition in query.target_definitions
        for relation in relations
    ]

    pairs = [
        Comparison(
            (
                f"Headword: {query.lemma}\nPart of speech: {query.pos}\n"
                f"Wiktionary: {render_definition(source_definition, mode)}\n"
                f"Required relation: {instructions.relations[relation]}"
            ),
            target_definition.glosses[-1],
        )
        for source_definition, target_definition, relation in candidates
    ]

    scores = scorer.score(pairs) if pairs else []
    if len(scores) != len(pairs) or not all(isfinite(score) for score in scores):
        raise ValueError("The scorer must return one finite score per candidate")

    return AlignmentResult(
        query,
        tuple(
            AlignmentScore(source_definition.id, target_definition.id, relation, score)
            for (source_definition, target_definition, relation), score in zip(
                candidates, scores, strict=True
            )
        ),
    )


def select_links(
    results: AlignmentResult,
    threshold: float,
) -> tuple[AlignmentScore, ...]:
    """
    Select partial translation matches or directed WordNet associations.

    Args:
        results: All candidate evidence.
        threshold: Model-specific boundary; accepted scores must be strictly greater.

    Returns:
        Accepted links, preserving source and target order.

    Raises:
        ValueError: If the threshold is not finite.
    """
    if not isfinite(threshold):
        raise ValueError("The threshold must be finite")

    if not results.scores:
        return ()

    if results.query.task == AlignmentTask.WORDNET:
        chosen_relations: dict[tuple[str, str], AlignmentScore] = {}

        for link in results.scores:
            key = (link.source_id, link.target_id)
            if key not in chosen_relations or link.score > chosen_relations[key].score:
                chosen_relations[key] = link

        return tuple(
            link for link in chosen_relations.values() if link.score > threshold
        )

    from scipy.optimize import linear_sum_assignment

    source_definitions = {
        source.id: position
        for position, source in enumerate(results.query.source_definitions)
    }

    target_definitions = {
        target.id: position
        for position, target in enumerate(results.query.target_definitions)
    }

    weights = [
        [0.0] * (len(target_definitions) + len(source_definitions))
        for _ in range(len(source_definitions))
    ]

    candidates: dict[tuple[int, int], AlignmentScore] = {}

    for link in results.scores:
        row, column = (
            source_definitions[link.source_id],
            target_definitions[link.target_id],
        )

        weights[row][column] = min(link.score - threshold, 0.0)

        if link.score > threshold:
            weights[row][column] = link.score - threshold
            candidates[row, column] = link

    rows, columns = linear_sum_assignment(weights, maximize=True)

    return tuple(
        candidates[int(row), int(column)]
        for row, column in zip(rows, columns, strict=True)
        if (int(row), int(column)) in candidates
    )
