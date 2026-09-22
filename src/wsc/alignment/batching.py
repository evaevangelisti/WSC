"""Prepare bounded inference batches and merge cached source decisions."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace

from ..models import Lemma, Sense, TranslationTable
from ..models.alignment import (
    AlignmentDecision,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    ModelRequest,
)
from .candidates import WordNetCandidates
from .decisions import validate_result
from .requests import build_request
from .tasks import TASK_HANDLERS, build_queries

type TaskCache = Mapping[str, Mapping[str, AlignmentDecision]]
type AlignmentCache = Mapping[AlignmentTask, TaskCache]


def _cached_decisions(
    query: AlignmentQuery,
    cache: AlignmentCache,
) -> tuple[AlignmentDecision, ...]:
    """
    Select cached decisions that remain valid for the current candidates.

    Args:
        query: Current source and candidate definitions.
        cache: Decisions indexed by task, query, and source.

    Returns:
        Valid cached decisions in current source order.
    """
    query_cache = cache.get(query.task, {}).get(query.alignment_id, {})
    decisions = tuple(
        query_cache[source.id]
        for source in query.source_definitions
        if source.id in query_cache
    )

    if not decisions:
        return ()

    source_ids = {decision.source_id for decision in decisions}
    cached_query = replace(
        query,
        source_definitions=tuple(
            source for source in query.source_definitions if source.id in source_ids
        ),
    )

    try:
        validate_result(AlignmentResult(cached_query, decisions))
    except ValueError:
        return ()

    return decisions


@dataclass(slots=True)
class PreparedQuery:
    """
    Hold cached decisions and any remaining model query.

    Attributes:
        query: Complete current query.
        decisions: Resolved decisions indexed by source.
        pending: Query containing only sources absent from the cache.
        response: Generated response for newly resolved sources.
    """

    query: AlignmentQuery

    decisions: dict[str, AlignmentDecision]
    pending: AlignmentQuery | None

    response: str = ""

    @property
    def complete(
        self,
    ) -> bool:
        """Return whether every current source has a decision."""
        return len(self.decisions) == len(self.query.source_definitions)

    def merge(
        self,
        result: AlignmentResult,
    ) -> None:
        """
        Merge generated decisions and validate the complete association graph.

        Args:
            result: Decisions generated for pending sources.

        Raises:
            ValueError: If cached and generated decisions conflict.
        """
        combined_decisions = {
            **self.decisions,
            **{decision.source_id: decision for decision in result.decisions},
        }

        decisions = tuple(
            combined_decisions[source.id]
            for source in self.query.source_definitions
            if source.id in combined_decisions
        )

        merged = AlignmentResult(self.query, decisions, result.response)
        validate_result(merged)

        self.decisions = combined_decisions
        self.pending = None

        self.response = result.response

    def result(
        self,
    ) -> AlignmentResult | None:
        """
        Build the resolved result while preserving current source order.

        Returns:
            The resolved sources and decisions, or None when no source is resolved.
        """
        sources = tuple(
            source
            for source in self.query.source_definitions
            if source.id in self.decisions
        )

        if not sources:
            return None

        query = replace(self.query, source_definitions=sources)
        decisions = tuple(self.decisions[source.id] for source in sources)

        return AlignmentResult(query, decisions, self.response)


@dataclass(slots=True)
class PreparedLemma:
    """
    Hold one copied entry and its prepared alignment queries.

    Attributes:
        lemma: Original collected entry.
        senses: Sense copies receiving accepted associations.
        translation_tables: Tables retained until translation succeeds.
        queries: Cached and pending work in task order.
    """

    lemma: Lemma
    senses: dict[str, Sense]
    translation_tables: tuple[TranslationTable, ...]
    queries: list[PreparedQuery] = field(default_factory=list)


@dataclass(slots=True)
class PreparedBatch:
    """
    Hold copied entries and prompts for one inference call.

    Attributes:
        lemmas: Buffered entries in collection order.
        requests: Prompts requiring model inference.
        queries: Prepared query corresponding to each prompt.
    """

    lemmas: list[PreparedLemma]
    requests: list[ModelRequest] = field(default_factory=list)
    queries: list[PreparedQuery] = field(default_factory=list)


def _prepare_query(
    query: AlignmentQuery,
    cache: AlignmentCache,
) -> PreparedQuery:
    """
    Combine reusable decisions with sources requiring inference.

    Args:
        query: Complete current alignment query.
        cache: Persisted source decisions.

    Returns:
        Prepared cached and pending work.
    """
    cached_decisions = _cached_decisions(query, cache)

    decisions = {decision.source_id: decision for decision in cached_decisions}
    missing_decisions = tuple(
        source for source in query.source_definitions if source.id not in decisions
    )

    targets = query.target_definitions

    if TASK_HANDLERS[query.task].one_to_one:
        assigned = {
            link.target_id for decision in cached_decisions for link in decision.links
        }

        targets = tuple(target for target in targets if target.id not in assigned)

    if missing_decisions and not targets:
        decisions.update(
            (source.id, AlignmentDecision(source.id)) for source in missing_decisions
        )

        missing_decisions = ()

    pending_decisions = (
        replace(query, source_definitions=missing_decisions, target_definitions=targets)
        if missing_decisions
        else None
    )

    return PreparedQuery(query, decisions, pending_decisions)


def prepare_batch(
    lemmas: Iterator[Lemma],
    candidates: WordNetCandidates,
    tasks: tuple[AlignmentTask, ...],
    mode: GlossMode,
    prompts: AlignmentPrompts,
    batch_size: int,
    cache: AlignmentCache,
) -> PreparedBatch | None:
    """
    Read entries until one bounded inference batch is ready.

    Args:
        lemmas: Remaining collected entries.
        candidates: WordNet candidate index.
        tasks: Requested alignment resources.
        mode: Wiktionary definition representation.
        prompts: Task prompt templates.
        batch_size: Maximum prompts or entries in the batch.
        cache: Persisted source decisions.

    Returns:
        Prepared batch, or None once entries are exhausted.
    """
    batch = PreparedBatch([])

    for lemma in lemmas:
        prepared_lemma = PreparedLemma(
            lemma,
            {sense.id: replace(sense) for sense in lemma.senses},
            lemma.translation_tables,
        )

        for task in tasks:
            for query in build_queries(lemma, task, candidates):
                prepared_query = _prepare_query(query, cache)
                prepared_lemma.queries.append(prepared_query)

                if prepared_query.pending is not None:
                    batch.requests.append(
                        build_request(prepared_query.pending, mode, prompts),
                    )
                    batch.queries.append(prepared_query)

        batch.lemmas.append(prepared_lemma)

        if len(batch.requests) >= batch_size or len(batch.lemmas) >= batch_size:
            break

    return batch if batch.lemmas else None
