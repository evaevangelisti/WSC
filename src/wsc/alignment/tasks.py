"""Define resource-specific query construction and association application."""

from collections.abc import Iterator
from typing import Protocol

from ..constants import TRANSLATION_RELATION
from ..models import Lemma, Sense, WordNetAlignment, WordNetRelation
from ..models.alignment import AlignmentLink, AlignmentQuery, AlignmentTask, Definition
from .candidates import WordNetCandidates


def query_id(
    task: AlignmentTask,
    identifier: str,
) -> str:
    """Build a task-scoped identifier for an alignment query."""
    return f"{task.value}:{identifier}"


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
    Collect sense definitions and their synonyms.

    Args:
        lemma: Collected entry.

    Returns:
        Definitions in source order.
    """
    return tuple(
        Definition(sense.id, sense.glosses, sense.synonyms) for sense in lemma.senses
    )


class TranslationHandler:
    """Align translation groups with distinct Wiktionary senses."""

    relations: tuple[str, ...] = (TRANSLATION_RELATION,)
    one_to_one: bool = True

    def queries(
        self,
        lemma: Lemma,
        candidates: WordNetCandidates,
    ) -> Iterator[AlignmentQuery]:
        """
        Construct one translation query per entry.

        Args:
            lemma: Entry containing translation groups.
            candidates: Shared resource index.

        Yields:
            The entry's complete senses and translation groups.
        """
        del candidates

        sources = build_definitions(lemma)

        if lemma.translation_tables and sources:
            yield AlignmentQuery(
                AlignmentTask.TRANSLATIONS,
                query_id(AlignmentTask.TRANSLATIONS, lemma.id),
                lemma.id,
                lemma.lemma,
                lemma.pos,
                sources,
                tuple(
                    Definition(table.id, (table.gloss,))
                    for table in lemma.translation_tables
                ),
            )

    def apply(
        self,
        lemma: Lemma,
        senses: dict[str, Sense],
        query: AlignmentQuery,
        links: tuple[AlignmentLink, ...],
    ) -> None:
        """
        Transfer accepted translation groups into senses.

        Args:
            lemma: Original translation dictionaries.
            senses: Copies receiving translations.
            query: Translation group identities.
            links: One-to-one associations.
        """
        for source in query.source_definitions:
            senses[source.id].translations = {}

        tables = {table.id: table for table in lemma.translation_tables}

        for link in links:
            senses[link.source_id].translations = dict(
                tables[link.target_id].translations
            )


class WordNetHandler:
    """Align each source sense with related WordNet concepts."""

    relations: tuple[str, ...] = tuple(WordNetRelation)
    one_to_one: bool = False

    def queries(
        self,
        lemma: Lemma,
        candidates: WordNetCandidates,
    ) -> Iterator[AlignmentQuery]:
        """
        Construct WordNet queries with lexical members as synonyms.

        Args:
            lemma: Entry supplying source senses.
            candidates: WordNet candidate index.

        Yields:
            One query for each source sense.
        """
        targets = tuple(
            Definition(
                synset.id,
                (synset.definition,),
                candidates.synonyms(lemma, synset),
            )
            for synset in candidates.candidates(lemma)
        )

        for source in build_definitions(lemma):
            yield AlignmentQuery(
                AlignmentTask.WORDNET,
                query_id(AlignmentTask.WORDNET, source.id),
                lemma.id,
                lemma.lemma,
                lemma.pos,
                (source,),
                targets,
            )

    def apply(
        self,
        lemma: Lemma,
        senses: dict[str, Sense],
        query: AlignmentQuery,
        links: tuple[AlignmentLink, ...],
    ) -> None:
        """
        Store synset identifiers and their directed relations.

        Args:
            lemma: Original collection entry.
            senses: Copies receiving WordNet associations.
            query: Source sense and candidate context.
            links: Accepted WordNet associations.
        """
        del lemma

        for source in query.source_definitions:
            senses[source.id].wordnet = tuple(
                WordNetAlignment(link.target_id, WordNetRelation(link.relation))
                for link in links
                if link.source_id == source.id
            )


TASK_HANDLERS: dict[AlignmentTask, AlignmentHandler] = {
    AlignmentTask.TRANSLATIONS: TranslationHandler(),
    AlignmentTask.WORDNET: WordNetHandler(),
}
"""Register query construction, relation constraints, and result application."""


def build_queries(
    lemma: Lemma,
    task: AlignmentTask,
    candidates: WordNetCandidates,
) -> Iterator[AlignmentQuery]:
    """
    Expose complete candidate sets without semantic filtering.

    Args:
        lemma: Collected entry.
        task: Resource to align.
        candidates: Cached WordNet candidate index.

    Yields:
        One translation entry or one task per WordNet source sense.
    """
    yield from TASK_HANDLERS[task].queries(lemma, candidates)
