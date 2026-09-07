"""
Reproducible samples and independent second annotation assignments.
"""

import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

from wsc.files import partial_file
from wsc.models.alignment import AlignmentQuery

SEED = 0
COUNT = 100
REPEATED_SHARE = 0.1
EXPORTS = ("lemminflect.jsonl", "spacy.jsonl", "stanza.jsonl")
HERE = Path(__file__).resolve().parent.parent


def sample_items[Item](items: Iterable[Item], count: int, seed: int) -> list[Item]:
    """
    Draw a uniform reservoir without retaining the complete corpus.

    Args:
        items: Population to sample.
        count: Requested number of unique items.
        seed: Reproducible random seed.

    Returns:
        A shuffled uniform sample.

    Raises:
        ValueError: If the population cannot supply the requested positive count.
    """
    if count < 1:
        raise ValueError("Sample count must be positive")
    rng = random.Random(seed)
    drawn: list[Item] = []
    for position, item in enumerate(items):
        if position < count:
            drawn.append(item)
        else:
            replacement = rng.randrange(position + 1)
            if replacement < count:
                drawn[replacement] = item
    if len(drawn) != count:
        raise ValueError(f"Requested {count} items; only {len(drawn)} available")
    rng.shuffle(drawn)

    return drawn


def write_json(path: Path, payload: object) -> None:
    """
    Write a complete research artifact atomically.

    Args:
        path: Artifact destination.
        payload: JSON-serializable content.
    """
    with partial_file(path) as partial:
        _ = partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def repeat_items[Item](items: Sequence[Item], seed: int) -> list[Item]:
    """
    Select ten percent for an independent second annotator.

    Args:
        items: Primary annotation sample.
        seed: Shared seed ensuring identical repeats across models.

    Returns:
        Independent assignments, rounded down to whole tasks.
    """
    return random.Random(seed).sample(list(items), int(len(items) * REPEATED_SHARE))


def split_queries(queries: Sequence[AlignmentQuery], seed: int) -> dict[str, str]:
    """
    Freeze development and test groups without splitting headwords.

    Args:
        queries: Complete sampled tasks before annotation.
        seed: Reproducible group shuffling seed.

    Returns:
        Task splits reserving approximately twenty percent of headwords for development.
    """
    lemmas = sorted({query.lemma.casefold() for query in queries})
    random.Random(seed).shuffle(lemmas)
    development = set(lemmas[: max(1, len(lemmas) // 5)])

    return {
        query.alignment_id: "development"
        if query.lemma.casefold() in development
        else "test"
        for query in queries
    }
