"""Apply generated lexical associations to collected senses."""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from logging import getLogger

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..constants import ALIGNMENT_BATCH_SIZE, DEFAULT_PROMPTS
from ..errors import InvalidModelResponseError
from ..models import Lemma
from ..models.alignment import (
    AlignmentDecision,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    LanguageModel,
    ModelOutcome,
)
from .batching import AlignmentCache, PreparedBatch, prepare_batch
from .candidates import WordNetCandidates
from .decisions import parse_response
from .requests import build_request
from .tasks import TASK_HANDLERS

_LOGGER = getLogger(__name__)


def _abstain(
    query: AlignmentQuery,
) -> AlignmentResult:
    """
    Build empty decisions when a query has nothing to compare.

    Args:
        query: Query without sources or candidates.

    Returns:
        One empty decision per source definition.
    """
    return AlignmentResult(
        query,
        tuple(AlignmentDecision(source.id) for source in query.source_definitions),
    )


def _parse_outcome(
    query: AlignmentQuery,
    outcome: ModelOutcome,
) -> AlignmentResult:
    """
    Validate one generated outcome against its query.

    Args:
        query: Sources and candidates supplied to the model.
        outcome: Generated text or its failure description.

    Returns:
        Validated alignment decisions.

    Raises:
        InvalidModelResponseError: If generation or validation failed.
    """
    if outcome.text is None:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}\n\n{outcome.error}"
        )

    try:
        return parse_response(query, outcome.text)
    except InvalidModelResponseError:
        raise
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Invalid model response for {query.alignment_id}\n"
            + f"{error}\n\nResponse\n{outcome.text}"
        ) from error


def align_query(
    query: AlignmentQuery,
    model: LanguageModel,
    mode: GlossMode = GlossMode.LAST,
    prompts: AlignmentPrompts = DEFAULT_PROMPTS,
) -> AlignmentResult:
    """
    Generate validated decisions for a complete alignment query.

    Args:
        query: Sources and available candidates.
        model: Language model generation boundary.
        mode: Wiktionary definition representation.
        prompts: Task prompt templates.

    Returns:
        Associations or explicit abstentions for every source.
    """
    if not query.target_definitions or not query.source_definitions:
        return _abstain(query)

    request = build_request(query, mode, prompts)

    try:
        response = model.generate(request)
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}\n\n{error}"
        ) from error

    return _parse_outcome(query, ModelOutcome(response))


class Aligner:
    """Align resources through language model decisions or cached results."""

    def __init__(
        self,
        model: LanguageModel | None,
        candidates: WordNetCandidates,
        tasks: Iterable[AlignmentTask] = (
            AlignmentTask.TRANSLATIONS,
            AlignmentTask.WORDNET,
        ),
        mode: GlossMode = GlossMode.LAST,
        prompts: AlignmentPrompts = DEFAULT_PROMPTS,
        batch_size: int = ALIGNMENT_BATCH_SIZE,
        model_loader: Callable[[], LanguageModel] | None = None,
    ) -> None:
        """
        Configure the model and requested alignment tasks.

        Args:
            model: Generation boundary, or None during cache replay.
            candidates: WordNet candidate index.
            tasks: Resources to align.
            mode: Wiktionary definition representation.
            prompts: Named task prompt templates.
            batch_size: Maximum prompts or entries prepared per batch.
            model_loader: Deferred model construction for partial reuse.
        """
        self._model: LanguageModel | None = model

        self._candidates: WordNetCandidates = candidates

        self._tasks: tuple[AlignmentTask, ...] = tuple(dict.fromkeys(tasks))

        self._mode: GlossMode = mode
        self._prompts: AlignmentPrompts = prompts

        self._batch_size: int = batch_size

        self._model_loader: Callable[[], LanguageModel] | None = model_loader

    def _report_failure(
        self,
        query: AlignmentQuery,
        error: InvalidModelResponseError,
    ) -> None:
        """
        Report one rejected response without stopping its batch.

        Args:
            query: Query whose response failed.
            error: Generation or validation failure.
        """
        with logging_redirect_tqdm(loggers=[getLogger("wsc")]):
            _LOGGER.warning("Skipped %s\n\n%s", query.alignment_id, error)

    def _generate_batch(
        self,
        batch: PreparedBatch,
    ) -> None:
        """
        Generate and validate every unresolved query in a batch.

        Args:
            batch: Entries and prompts prepared for inference.

        Raises:
            ValueError: If inference is required without a model.
        """
        if not batch.requests:
            return

        if self._model is None:
            if self._model_loader is None:
                raise ValueError("Uncached alignment requires a language model")

            self._model = self._model_loader()

        _LOGGER.info("Generating %s alignment prompts", len(batch.requests))

        outcomes = self._model.generate_batch(batch.requests)

        for prepared, outcome in zip(
            batch.queries,
            outcomes,
            strict=True,
        ):
            query = prepared.pending

            if query is None:
                continue

            try:
                prepared.merge(_parse_outcome(query, outcome))
            except (InvalidModelResponseError, ValueError) as error:
                failure = (
                    error
                    if isinstance(error, InvalidModelResponseError)
                    else InvalidModelResponseError(
                        f"Invalid combined response for {query.alignment_id}\n\n{error}"
                    )
                )

                self._report_failure(query, failure)

    def _apply_batch(
        self,
        batch: PreparedBatch,
        recorder: Callable[[AlignmentResult], None] | None,
    ) -> Iterator[Lemma]:
        """
        Apply resolved decisions to copied entries.

        Args:
            batch: Entries carrying resolved query results.
            recorder: Optional callback persisting decisions.

        Yields:
            Aligned entries in their original order.
        """
        for prepared in batch.lemmas:
            for query in prepared.queries:
                result = query.result()

                if result is None:
                    continue

                _LOGGER.debug(
                    "Aligned %s\n\n%s",
                    query.query.alignment_id,
                    result.response,
                )

                if recorder is not None:
                    recorder(result)

                TASK_HANDLERS[query.query.task].apply(
                    prepared.lemma,
                    prepared.senses,
                    result.query,
                    result.links,
                )

                if query.query.task == AlignmentTask.TRANSLATIONS and query.complete:
                    prepared.translation_tables = ()

            yield replace(
                prepared.lemma,
                senses=list(prepared.senses.values()),
                translation_tables=prepared.translation_tables,
            )

    def _batches(
        self,
        lemmas: Iterable[Lemma],
        cache: AlignmentCache,
    ) -> Iterator[PreparedBatch]:
        """
        Prepare the next batch while the current batch is inferred.

        One worker prepares at most one batch ahead, limiting additional memory.

        Args:
            lemmas: Original collected entries.
            cache: Persisted decisions indexed by task, query, and source.

        Yields:
            Prepared batches in collection order.
        """
        remaining = iter(lemmas)

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wsc-prompts",
        ) as executor:
            future = executor.submit(
                prepare_batch,
                remaining,
                self._candidates,
                self._tasks,
                self._mode,
                self._prompts,
                self._batch_size,
                cache,
            )

            while (batch := future.result()) is not None:
                future = executor.submit(
                    prepare_batch,
                    remaining,
                    self._candidates,
                    self._tasks,
                    self._mode,
                    self._prompts,
                    self._batch_size,
                    cache,
                )

                yield batch

    def align(
        self,
        lemmas: Iterable[Lemma],
        *,
        cache: AlignmentCache | None = None,
        recorder: Callable[[AlignmentResult], None] | None = None,
    ) -> Iterator[Lemma]:
        """
        Stream aligned copies of collected entries.

        Args:
            lemmas: Original collected entries.
            cache: Persisted decisions indexed by task, query, and source.
            recorder: Optional callback persisting resolved decisions.

        Yields:
            Aligned entries with processed translation tables removed.
        """
        with tqdm(desc="Aligning senses", unit=" lemma") as progress:
            for batch in self._batches(lemmas, cache or {}):
                self._generate_batch(batch)

                for lemma in self._apply_batch(batch, recorder):
                    _ = progress.update()

                    yield lemma
