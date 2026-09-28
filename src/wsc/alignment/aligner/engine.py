"""
Apply generated lexical associations to collected senses.
"""

import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from logging import getLogger

from tqdm import tqdm

from ...constants import ALIGNMENT_BATCH_SIZE, DEFAULT_PROMPTS
from ...errors import InvalidModelResponseError
from ...models import Lemma
from ...models.alignment import (
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    LanguageModel,
)
from ..batching import AlignmentCache, PreparedBatch, prepare_batch
from ..candidates import SynsetCandidates
from ..tasks import TASK_HANDLERS
from .query import align_query, parse_outcome

__all__ = ["Aligner", "align_query"]

_LOGGER = getLogger(__package__)


class Aligner:
    """
    Align resources through language model decisions or cached results.
    """

    def __init__(
        self,
        model: LanguageModel | None,
        candidates: SynsetCandidates,
        tasks: Iterable[AlignmentTask] = (
            AlignmentTask.TRANSLATIONS,
            AlignmentTask.SYNSETS,
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
            candidates: Synset candidate index.
            tasks: Resources to align.
            mode: Wiktionary definition representation.
            prompts: Named task prompt templates.
            batch_size: Maximum prompts or entries prepared per batch.
            model_loader: Deferred model construction for partial reuse.
        """
        self._model: LanguageModel | None = model

        self._candidates: SynsetCandidates = candidates

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
        _LOGGER.debug("Skipped %s: %s", query.alignment_id, error)

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

        _LOGGER.debug("Generating %s alignment prompts", len(batch.requests))

        try:
            outcomes = self._model.generate_batch(batch.requests)
        finally:
            if (error := sys.exception()) is not None:
                first = batch.queries[0].query.alignment_id
                last = batch.queries[-1].query.alignment_id

                interval = first if first == last else f"{first} through {last}"

                error.add_note(
                    f"Failed alignment batch ({len(batch.requests)}): {interval}"
                )

        for prepared, outcome in zip(
            batch.queries,
            outcomes,
            strict=True,
        ):
            query = prepared.pending

            if query is None:
                continue

            try:
                prepared.merge(parse_outcome(query, outcome))
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
        prepare = partial(
            prepare_batch,
            iter(lemmas),
            self._candidates,
            self._tasks,
            self._mode,
            self._prompts,
            self._batch_size,
            cache,
        )

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wsc-prompts",
        ) as executor:
            future = executor.submit(prepare)

            while (batch := future.result()) is not None:
                future = executor.submit(prepare)

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
