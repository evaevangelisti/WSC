"""Construct translation queries and apply accepted associations."""

from collections.abc import Iterator

from ...constants import TRANSLATION_RELATION
from ...identifiers import query_id
from ...models import Lemma, Sense
from ...models.alignment import AlignmentLink, AlignmentQuery, AlignmentTask, Definition
from ..candidates import WordNetCandidates
from .base import build_definitions


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
        Transfer accepted translation tables into senses.

        Args:
            lemma: Original translation dictionaries.
            senses: Copies receiving translation tables.
            query: Translation group identities.
            links: One-to-one associations.
        """
        for source in query.source_definitions:
            senses[source.id].translation_table = None

        tables = {table.id: table for table in lemma.translation_tables}

        for link in links:
            senses[link.source_id].translation_table = tables[link.target_id]
