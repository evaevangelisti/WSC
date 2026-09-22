"""Define task handlers and shared Wiktionary sense construction."""

from collections.abc import Iterator
from typing import Protocol

from ...models import Lemma, Sense
from ...models.alignment import AlignmentLink, AlignmentQuery, Definition
from ..candidates import WordNetCandidates


class AlignmentHandler(Protocol):
    """Define the extension points for an alignment task."""

    relations: tuple[str, ...]
    one_to_one: bool

    def queries(
        self,
        lemma: Lemma,
        candidates: WordNetCandidates,
    ) -> Iterator[AlignmentQuery]:
        """
        Construct resource-specific candidate sets.

        Args:
            lemma: Collected entry.
            candidates: WordNet candidate index.

        Yields:
            Complete alignment queries.
        """
        ...

    def apply(
        self,
        lemma: Lemma,
        senses: dict[str, Sense],
        query: AlignmentQuery,
        links: tuple[AlignmentLink, ...],
    ) -> None:
        """
        Apply accepted associations to copied senses.

        Args:
            lemma: Original collection entry.
            senses: Copies receiving the associations.
            query: Complete candidate context.
            links: Accepted associations.
        """
        ...


def build_definitions(
    lemma: Lemma,
) -> tuple[Definition, ...]:
    """
    Collect sense definitions and their lexical context.

    Args:
        lemma: Collected entry.

    Returns:
        Definitions in source order.
    """
    return tuple(
        Definition(
            sense.id,
            sense.glosses,
            synonyms=sense.synonyms,
            tags=sense.tags,
            topics=sense.topics,
        )
        for sense in lemma.senses
    )
