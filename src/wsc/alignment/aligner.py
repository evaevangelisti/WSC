"""
Apply semantic associations to collected senses.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace

from ..constants import DEFAULT_INSTRUCTIONS
from ..models import Lemma, Sense, WordNetAlignment, WordNetRelation
from ..models.alignment import (
    AlignmentInstructions,
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    GlossMode,
    Scorer,
)
from .candidates import WordNetCandidates, build_queries
from .scoring import score_query, select_links


class Aligner:
    """Align collected resources using model inference or persisted candidate scores."""

    def __init__(
        self,
        scorer: Scorer | None,
        candidates: WordNetCandidates,
        thresholds: Mapping[AlignmentTask, float],
        mode: GlossMode = GlossMode.LAST,
        instructions: AlignmentInstructions = DEFAULT_INSTRUCTIONS,
    ) -> None:
        """
        Configure semantic scoring and resource-specific abstention.

        Args:
            scorer: Semantic model, or None when replaying cached scores.
            candidates: WordNet candidate index.
            thresholds: Requested resources and their raw score thresholds.
            mode: Source definition representation.
            instructions: Semantic hypotheses matching the configured scorer.
        """
        self._scorer: Scorer | None = scorer

        self._candidates: WordNetCandidates = candidates

        self._thresholds: dict[AlignmentTask, float] = dict(thresholds)
        self._mode: GlossMode = mode
        self._instructions: AlignmentInstructions = instructions

    def _evaluate(
        self,
        query: AlignmentQuery,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]],
    ) -> AlignmentResult:
        """
        Retrieve compatible cached evidence or evaluate the semantic model.

        Args:
            query: Current candidate definitions.
            cached_results: Cached result streams selected explicitly by the caller.

        Returns:
            Complete candidate scores.

        Raises:
            ValueError: If cached inputs differ or inference has no model.
        """
        if query.task in cached_results:
            result = next(cached_results[query.task], None)
            if result is None or result.query != query:
                raise ValueError(f"Cached candidates differ for {query.alignment_id}")

            return result

        if self._scorer is None:
            raise ValueError("Uncached alignment requires a semantic scorer")

        return score_query(query, self._scorer, self._mode, self._instructions)

    @staticmethod
    def _apply_links(
        lemma: Lemma,
        senses: dict[str, Sense],
        query: AlignmentQuery,
        links: tuple[AlignmentScore, ...],
    ) -> None:
        """
        Transfer accepted associations into copied senses.

        Args:
            lemma: Original collection entry supplying translations.
            senses: Copied senses receiving the associations.
            query: Candidate definitions and alignment resource.
            links: Associations accepted after thresholding and assignment.
        """
        match query.task:
            case AlignmentTask.TRANSLATIONS:
                for source_definition in query.source_definitions:
                    senses[source_definition.id].translations = {}

                target_definitions = {
                    target.id: target.glosses[-1] for target in query.target_definitions
                }

                for link in links:
                    senses[link.source_id].translations = dict(
                        lemma.translations[target_definitions[link.target_id]]
                    )

            case AlignmentTask.WORDNET:
                senses[query.source_definitions[0].id].wordnet = tuple(
                    WordNetAlignment(link.target_id, WordNetRelation(link.relation))
                    for link in links
                )

    def align(
        self,
        lemmas: Iterable[Lemma],
        *,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]] | None = None,
        scores: Callable[[AlignmentResult], None] | None = None,
    ) -> Iterator[Lemma]:
        """
        Stream aligned copies while retaining the original collection.

        Args:
            lemmas: Collected entries.
            cached_results: Optional resource-specific score streams for replay.
            scores: Optional sink retaining unfiltered semantic evidence.

        Yields:
            Aligned entries with processed lemma-level translations removed.

        Raises:
            ValueError: If cached evidence differs from the collection.
        """
        streams = cached_results or {}

        for lemma in lemmas:
            senses = {
                sense.id: replace(sense, translations=dict(sense.translations))
                for sense in lemma.senses
            }

            for task, threshold in self._thresholds.items():
                for query in build_queries(lemma, task, self._candidates):
                    result = self._evaluate(query, streams)

                    if scores is not None:
                        scores(result)

                    links = select_links(result, threshold)

                    self._apply_links(lemma, senses, query, links)

            yield replace(
                lemma,
                senses=list(senses.values()),
                translations=(
                    {}
                    if AlignmentTask.TRANSLATIONS in self._thresholds
                    else lemma.translations
                ),
            )

        for task, stream in streams.items():
            if next(stream, None) is not None:
                raise ValueError(f"Unused cached {task} scores remain")
