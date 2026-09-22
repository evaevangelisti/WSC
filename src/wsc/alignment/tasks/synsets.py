"""Construct generic synset queries and apply semantic associations."""

from collections.abc import Iterator

from ...identifiers import query_id
from ...models import Lemma, Sense, SynsetAlignment, SynsetRelation
from ...models.alignment import AlignmentLink, AlignmentQuery, AlignmentTask, Definition
from ..candidates import SynsetCandidates
from .base import build_definitions


class SynsetHandler:
    """Align each Wiktionary sense with supplied generic synsets."""

    relations: tuple[str, ...] = tuple(SynsetRelation)
    one_to_one: bool = False

    def queries(
        self,
        lemma: Lemma,
        candidates: SynsetCandidates,
    ) -> Iterator[AlignmentQuery]:
        """Construct synset queries with lexical members as synonyms.

        Args:
            lemma: Entry supplying source senses.
            candidates: Synset candidate index.

        Yields:
            One query containing all source senses and candidates.
        """
        targets = tuple(
            Definition(
                synset.id,
                synset.glosses,
                synonyms=candidates.synonyms(lemma, synset),
                examples=synset.examples,
                source=candidates.source(lemma, synset),
            )
            for synset in candidates.candidates(lemma)
        )

        sources = build_definitions(lemma)

        if sources:
            yield AlignmentQuery(
                AlignmentTask.SYNSETS,
                query_id(AlignmentTask.SYNSETS, lemma.id),
                lemma.id,
                lemma.lemma,
                lemma.pos,
                sources,
                targets,
            )

    def apply(
        self,
        lemma: Lemma,
        senses: dict[str, Sense],
        query: AlignmentQuery,
        links: tuple[AlignmentLink, ...],
    ) -> None:
        """Store synset identifiers, sources, and their relations.

        Args:
            lemma: Original collection entry.
            senses: Copies receiving synset associations.
            query: Source sense and candidate context.
            links: Accepted synset associations.
        """
        del lemma

        target_sources = {
            target.id: target.source for target in query.target_definitions
        }

        for source in query.source_definitions:
            senses[source.id].synsets = tuple(
                SynsetAlignment(
                    link.target_id,
                    SynsetRelation(link.relation),
                    target_sources.get(link.target_id, ""),
                )
                for link in links
                if link.source_id == source.id
            )
