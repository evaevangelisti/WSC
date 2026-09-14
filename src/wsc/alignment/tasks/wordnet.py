"""Construct WordNet queries and apply directed semantic associations."""

from collections.abc import Iterator

from ...identifiers import query_id
from ...models import Lemma, Sense, WordNetAlignment, WordNetRelation
from ...models.alignment import AlignmentLink, AlignmentQuery, AlignmentTask, Definition
from ..candidates import WordNetCandidates
from .base import build_definitions


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
            One query containing all source senses and candidates.
        """
        targets = tuple(
            Definition(
                synset.id,
                (synset.definition,),
                candidates.synonyms(lemma, synset),
            )
            for synset in candidates.candidates(lemma)
        )

        sources = build_definitions(lemma)

        if sources:
            yield AlignmentQuery(
                AlignmentTask.WORDNET,
                query_id(AlignmentTask.WORDNET, lemma.id),
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
