"""
Link-level evaluation and development-only threshold selection.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import nextafter

from annotation.scripts.alignment import Judgement, Link

from wsc.alignment import select_links
from wsc.models.alignment import AlignmentResult


@dataclass(frozen=True, slots=True)
class Observation:
    """
    One determinate manual task compared with a model assignment.

    Attributes:
        gold: Human reference and optional qualitative notes.
        predicted: Selected model associations.
    """

    gold: Judgement
    predicted: frozenset[Link]


def evaluate(
    results: dict[str, AlignmentResult],
    judgements: Sequence[Judgement],
    threshold: float,
) -> list[Observation]:
    """
    Compare complete link sets on determinate, candidate-present tasks.

    Args:
        results: Complete semantic evidence indexed by task identifier.
        judgements: Consolidated manual references.
        threshold: Model-specific abstention boundary.

    Returns:
        Independent observations excluding uncertain and candidate-missing tasks.

    Raises:
        ValueError: If predictions lack a reference task or use different candidates.
    """
    observations: list[Observation] = []
    for judgement in judgements:
        identifier = judgement.query.alignment_id
        if identifier not in results or results[identifier].query != judgement.query:
            raise ValueError(
                f"Prediction context differs from annotation: {identifier}"
            )
        if judgement.status not in {"matched", "no_match"}:
            continue
        selected = select_links(results[identifier], threshold)
        observations.append(
            Observation(
                judgement,
                frozenset(
                    (link.source_id, link.target_id, link.relation) for link in selected
                ),
            )
        )

    return observations


def _ratio(numerator: int, denominator: int) -> float | None:
    """
    Preserve undefined metrics when their denominator is zero.

    Args:
        numerator: Successful outcomes.
        denominator: Eligible outcomes.

    Returns:
        Fraction, or None when undefined.
    """
    return numerator / denominator if denominator else None


def summarize(observations: Sequence[Observation]) -> dict[str, float | None]:
    """
    Aggregate association quality separately from source and target coverage.

    Args:
        observations: Unique, determinate manual tasks.

    Returns:
        Link metrics and task-level diagnostics with explicit undefined values.
    """
    correct = sum(len(item.predicted & item.gold.links) for item in observations)
    predicted = sum(len(item.predicted) for item in observations)
    expected = sum(len(item.gold.links) for item in observations)
    exact = sum(item.predicted == item.gold.links for item in observations)
    no_match = [item for item in observations if item.gold.status == "no_match"]
    source_count = sum(len(item.gold.query.source_definitions) for item in observations)
    target_count = sum(len(item.gold.query.target_definitions) for item in observations)
    source_coverage = sum(
        len({source for source, _, _ in item.predicted}) for item in observations
    )
    target_coverage = sum(
        len({target for _, target, _ in item.predicted}) for item in observations
    )
    grouped: defaultdict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        grouped[item.gold.query.lemma.casefold()].append(item)
    macro: list[float] = []
    for items in grouped.values():
        hits = sum(len(item.predicted & item.gold.links) for item in items)
        total = sum(len(item.predicted) + len(item.gold.links) for item in items)
        if total:
            macro.append(2 * hits / total)

    return {
        "Precision": _ratio(correct, predicted),
        "Recall": _ratio(correct, expected),
        "F1": _ratio(2 * correct, predicted + expected),
        "Macro F1": sum(macro) / len(macro) if macro else None,
        "Source coverage": _ratio(source_coverage, source_count),
        "Target coverage": _ratio(target_coverage, target_count),
        "Exact tasks": _ratio(exact, len(observations)),
        "No-match accuracy": _ratio(
            sum(bool(item.predicted) == bool(item.gold.links) for item in observations),
            len(observations),
        ),
        "No-match recall": _ratio(
            sum(not item.predicted for item in no_match), len(no_match)
        ),
    }


def threshold_curve(
    results: dict[str, AlignmentResult],
    judgements: Sequence[Judgement],
) -> list[tuple[float, dict[str, float | None]]]:
    """
    Sweep development evidence without reading test labels.

    Args:
        results: Candidate scores for the shared sample.
        judgements: Development references only.

    Returns:
        At most 102 thresholds and their measured metrics.

    Raises:
        ValueError: If development has no determinate manually annotated tasks.
    """
    determinate = [
        item for item in judgements if item.status in {"matched", "no_match"}
    ]
    if not determinate:
        raise ValueError(
            "Threshold selection requires determinate development annotations"
        )
    scores = sorted(
        {
            score.score
            for item in determinate
            for score in results[item.query.alignment_id].scores
        }
    )
    thresholds = (
        [0.0]
        if not scores
        else [
            nextafter(scores[0], float("-inf")),
            *(
                scores[index]
                for index in sorted(
                    {position * (len(scores) - 1) // 100 for position in range(101)}
                )
            ),
        ]
    )

    return [
        (threshold, summarize(evaluate(results, determinate, threshold)))
        for threshold in thresholds
    ]


def select_threshold(
    curve: Sequence[tuple[float, dict[str, float | None]]],
    minimum_precision: float,
) -> tuple[float, bool]:
    """
    Maximize development recall subject to the requested empirical precision.

    Args:
        curve: Development-only operating points.
        minimum_precision: Required empirical precision between zero and one.

    Returns:
        Chosen threshold and whether a nonempty operating point satisfies the target.

    Raises:
        ValueError: If precision is outside its probability range.
    """
    if not 0 <= minimum_precision <= 1:
        raise ValueError("Minimum precision must lie between zero and one")
    eligible = [
        (threshold, metrics)
        for threshold, metrics in curve
        if metrics["Precision"] is not None
        and metrics["Precision"] >= minimum_precision
    ]
    if not eligible:
        return curve[-1][0], False
    selected = max(
        eligible,
        key=lambda point: (
            point[1]["Recall"] or 0.0,
            point[1]["Precision"] or 0.0,
            point[0],
        ),
    )

    return selected[0], True


def confidence_intervals(
    observations: Sequence[Observation],
    seed: int,
    repetitions: int = 1000,
) -> dict[str, tuple[float, float] | None]:
    """
    Bootstrap headwords to retain dependence between related senses.

    Args:
        observations: Held-out task observations.
        seed: Bootstrap random seed.
        repetitions: Number of resampled headword collections.

    Returns:
        Percentile intervals for defined metrics.
    """
    grouped: defaultdict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        grouped[item.gold.query.lemma.casefold()].append(item)
    pools = list(grouped.values())
    rng = random.Random(seed)
    values: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sample = [item for group in rng.choices(pools, k=len(pools)) for item in group]
        for metric, value in summarize(sample).items():
            if value is not None:
                values[metric].append(value)

    return {
        metric: (
            sorted(values[metric])[int((len(values[metric]) - 1) * 0.025)],
            sorted(values[metric])[int((len(values[metric]) - 1) * 0.975)],
        )
        if values[metric]
        else None
        for metric in summarize(observations)
    }
