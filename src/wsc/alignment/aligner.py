"""Apply generated lexical associations to collected senses."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from logging import getLogger

from tqdm.contrib.logging import logging_redirect_tqdm

from ..constants import ALIGNMENT_BATCH_SIZE, DEFAULT_PROMPTS
from ..errors import InvalidModelResponseError
from ..models import Lemma, Sense, TranslationTable
from ..models.alignment import (
    AlignmentDecision,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    LanguageModel,
    ModelOutcome,
    ModelRequest,
)
from ..progress import nested_position, progress_bar
from .candidates import WordNetCandidates
from .decisions import parse_response, validate_result
from .requests import build_request
from .tasks import TASK_HANDLERS, build_queries

_LOGGER = getLogger(__name__)


def count_prompts(
    lemmas: Iterable[Lemma],
    tasks: Iterable[AlignmentTask],
    candidates: WordNetCandidates,
) -> int:
    """
    Count requests requiring inference using the alignment query filters.

    Args:
        lemmas: Collected entries to inspect without retaining them.
        tasks: Resources selected for inference.
        candidates: WordNet candidate index used for alignment.

    Returns:
        Number of queries with both source and target definitions.
    """
    selected = tuple(dict.fromkeys(tasks))
    return sum(
        bool(query.source_definitions and query.target_definitions)
        for lemma in lemmas
        for task in selected
        for query in build_queries(lemma, task, candidates)
    )


def abstain(
    query: AlignmentQuery,
) -> AlignmentResult:
    """
    Abstain for every source when the model has nothing to decide.

    Args:
        query: Query without sources or without candidates.

    Returns:
        One empty decision per source.
    """
    return AlignmentResult(
        query,
        tuple(AlignmentDecision(source.id) for source in query.source_definitions),
    )


def decode_outcome(
    query: AlignmentQuery,
    outcome: ModelOutcome,
) -> AlignmentResult:
    """
    Validate one generated response against its query.

    Args:
        query: Sources and candidates supplied to the model.
        outcome: Generated text, or the reason generation failed.

    Returns:
        Associations or explicit abstentions for every source.

    Raises:
        InvalidModelResponseError: If generation failed or the response is invalid.
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
        mode: Wiktionary gloss representation.
        prompts: Task prompt templates.

    Returns:
        Associations or explicit abstentions for every source.
    """
    if not query.target_definitions or not query.source_definitions:
        return abstain(query)

    request = build_request(query, mode, prompts)

    try:
        response = model.generate(request)
    except ValueError as error:
        raise InvalidModelResponseError(
            f"Model generation failed for {query.alignment_id}\n\n{error}"
        ) from error

    return decode_outcome(query, ModelOutcome(response))


@dataclass(slots=True)
class _BufferedLemma:
    """
    Hold a buffered entry between prompt construction and result application.

    Attributes:
        lemma: Original collected entry.
        senses: Sense copies receiving the associations.
        translation_tables: Tables retained until translations are aligned.
        queries: Queries generated for the entry, in task order.
        results: Resolved result per query, or None when the query was skipped.
    """

    lemma: Lemma
    senses: dict[str, Sense]
    translation_tables: tuple[TranslationTable, ...]
    queries: list[AlignmentQuery] = field(default_factory=list)
    results: list[AlignmentResult | None] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedBatch:
    """
    Hold one batch whose prompts are built and awaiting generation.

    Attributes:
        entries: Buffered entries of the batch, in collection order.
        requests: Prompts requiring generation, in submission order.
        positions: Entry and query index receiving each generated outcome.
    """

    entries: list[_BufferedLemma]
    requests: list[ModelRequest] = field(default_factory=list)
    positions: list[tuple[_BufferedLemma, int]] = field(default_factory=list)


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
    ) -> None:
        """
        Configure the model and requested alignment tasks.

        Args:
            model: Generation boundary, or None during cache replay.
            candidates: WordNet candidate index.
            tasks: Resources to align.
            mode: Wiktionary definition representation.
            prompts: Named task prompt templates.
            batch_size: Prompts built before each inference pass.
        """
        self._model: LanguageModel | None = model

        self._candidates: WordNetCandidates = candidates

        self._tasks: tuple[AlignmentTask, ...] = tuple(dict.fromkeys(tasks))

        self._mode: GlossMode = mode
        self._prompts: AlignmentPrompts = prompts
        self._batch_size: int = batch_size

    def _skip(
        self,
        query: AlignmentQuery,
        error: InvalidModelResponseError,
    ) -> None:
        """
        Report a query left unaligned without interrupting the batch.

        Args:
            query: Query whose decisions were rejected.
            error: Generation or validation failure.
        """
        with logging_redirect_tqdm(loggers=[getLogger("wsc")]):
            _LOGGER.warning("Skipped %s\n\n%s", query.alignment_id, error)

    def _buffer(
        self,
        lemma: Lemma,
    ) -> _BufferedLemma:
        """
        Copy an entry and generate every query it contributes.

        Args:
            lemma: Original collected entry.

        Returns:
            The entry with its queries awaiting resolution.
        """
        senses = {
            sense.id: replace(sense, translations=dict(sense.translations))
            for sense in lemma.senses
        }

        queries = [
            query
            for task in self._tasks
            for query in build_queries(lemma, task, self._candidates)
        ]

        return _BufferedLemma(
            lemma,
            senses,
            lemma.translation_tables,
            queries,
            [None] * len(queries),
        )

    def _replay(
        self,
        query: AlignmentQuery,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]],
    ) -> AlignmentResult:
        """
        Retrieve the cached decisions recorded for a query.

        Args:
            query: Current candidate context.
            cached_results: Explicitly selected cached streams.

        Returns:
            Validated decisions for the query.

        Raises:
            ValueError: If the cache context differs from the query.
        """
        result = next(cached_results[query.task], None)

        if result is None or result.query != query:
            raise ValueError(f"Cached candidates differ for {query.alignment_id}")

        validate_result(result)

        return result

    def _prepare(
        self,
        lemmas: Iterator[Lemma],
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]],
    ) -> _PreparedBatch | None:
        """
        Read entries until the configured number of prompts is built.

        Args:
            lemmas: Remaining collected entries.
            cached_results: Explicitly selected cached streams.

        Returns:
            The next batch, or None once the collection is exhausted.

        Raises:
            ValueError: If cached decisions differ from the collection.
        """
        batch = _PreparedBatch([])

        for lemma in lemmas:
            buffered = self._buffer(lemma)
            requests: list[ModelRequest] = []

            for index, query in enumerate(buffered.queries):
                if query.task in cached_results:
                    buffered.results[index] = self._replay(query, cached_results)

                    continue

                if not query.target_definitions or not query.source_definitions:
                    buffered.results[index] = abstain(query)

                    continue

                requests.append(build_request(query, self._mode, self._prompts))
                batch.positions.append((buffered, index))

            batch.entries.append(buffered)
            batch.requests.extend(requests)

            if len(batch.requests) >= self._batch_size:
                break

        return batch if batch.entries else None

    def _batches(
        self,
        lemmas: Iterable[Lemma],
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]],
    ) -> Iterator[_PreparedBatch]:
        """
        Prepare one batch of prompts while the previous one is generated.

        A worker thread reads ahead by a single batch, so prompt construction
        overlaps inference while only the running and the upcoming batch are
        held in memory.

        Args:
            lemmas: Original collected entries.
            cached_results: Explicitly selected cached streams.

        Yields:
            Batches of buffered entries whose prompts are ready to generate.
        """
        remaining = iter(lemmas)

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wsc-prompts",
        ) as pool:
            upcoming = pool.submit(self._prepare, remaining, cached_results)

            while (batch := upcoming.result()) is not None:
                upcoming = pool.submit(self._prepare, remaining, cached_results)

                yield batch

    def _generate(
        self,
        batch: _PreparedBatch,
    ) -> None:
        """
        Generate the prepared prompts of a batch in a single inference pass.

        Args:
            batch: Prepared entries whose results are filled in place.

        Raises:
            ValueError: If generation is required without a language model.
        """
        if not batch.requests:
            return

        if self._model is None:
            raise ValueError("Uncached alignment requires a language model")

        _LOGGER.info(
            "Generating %d prompts for %d entries",
            len(batch.requests),
            len(batch.entries),
        )

        outcomes = self._model.generate_many(batch.requests)

        for (buffered, index), outcome in zip(
            batch.positions,
            outcomes,
            strict=True,
        ):
            query = buffered.queries[index]

            try:
                buffered.results[index] = decode_outcome(query, outcome)
            except InvalidModelResponseError as error:
                self._skip(query, error)

    def _apply(
        self,
        batch: _PreparedBatch,
        recorder: Callable[[AlignmentResult], None] | None,
    ) -> Iterator[Lemma]:
        """
        Apply the decisions of a generated batch to its buffered entries.

        Args:
            batch: Entries whose results are resolved.
            recorder: Optional callback persisting generated decisions.

        Yields:
            Aligned entries with processed lemma-level translations removed.
        """
        for buffered in batch.entries:
            for query, result in zip(buffered.queries, buffered.results, strict=True):
                if result is None:
                    continue

                _LOGGER.debug("Aligned %s\n\n%s", query.alignment_id, result.response)

                if recorder is not None:
                    recorder(result)

                TASK_HANDLERS[query.task].apply(
                    buffered.lemma,
                    buffered.senses,
                    query,
                    result.links,
                )

                if query.task == AlignmentTask.TRANSLATIONS:
                    buffered.translation_tables = ()

            yield replace(
                buffered.lemma,
                senses=list(buffered.senses.values()),
                translation_tables=buffered.translation_tables,
            )

    def align(
        self,
        lemmas: Iterable[Lemma],
        *,
        total: int | None = None,
        total_prompts: int | None = None,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]] | None = None,
        recorder: Callable[[AlignmentResult], None] | None = None,
    ) -> Iterator[Lemma]:
        """
        Stream aligned copies of collected entries.

        Every prompt of a batch is built before the batch is sent to the model,
        and the next batch is prepared while the current one is generated, so
        inference runs once per batch without buffering the whole collection.

        Args:
            lemmas: Original collected entries.
            total: Expected number of entries, reported as the estimated time.
            total_prompts: Expected inference requests across all batches.
            cached_results: Task-specific decision streams for replay.
            recorder: Optional callback persisting generated decisions.

        Yields:
            Aligned entries with processed lemma-level translations removed.

        Raises:
            ValueError: If cached decisions differ from the collection.
        """
        streams = cached_results or {}

        with (
            progress_bar("Total prompts", " prompt", total_prompts) as prompts_progress,
            progress_bar(
                "Aligning senses", " lemma", total, position=2 * nested_position()
            ) as progress,
        ):
            for batch in self._batches(lemmas, streams):
                self._generate(batch)
                _ = prompts_progress.update(len(batch.requests))

                for lemma in self._apply(batch, recorder):
                    _ = progress.update()

                    yield lemma

        for task, stream in streams.items():
            if next(stream, None) is not None:
                raise ValueError(f"Unused cached {task} decisions remain")
