"""Apply generated lexical associations to collected senses."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from logging import getLogger

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..constants import DEFAULT_PROMPTS
from ..errors import InvalidModelResponseError
from ..models import Lemma
from ..models.alignment import (
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    LanguageModel,
)
from .candidates import WordNetCandidates
from .decisions import (
    align_query,
    validate_result,
)
from .tasks import TASK_HANDLERS, build_queries

_LOGGER = getLogger(__name__)


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
    ) -> None:
        """
        Configure the model and requested alignment tasks.

        Args:
            model: Generation boundary, or None during cache replay.
            candidates: WordNet candidate index.
            tasks: Resources to align.
            mode: Wiktionary definition representation.
            prompts: Named task prompt templates.
        """
        self._model: LanguageModel | None = model

        self._candidates: WordNetCandidates = candidates

        self._tasks: tuple[AlignmentTask, ...] = tuple(dict.fromkeys(tasks))

        self._mode: GlossMode = mode
        self._prompts: AlignmentPrompts = prompts

    def _evaluate(
        self,
        query: AlignmentQuery,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]],
    ) -> AlignmentResult:
        """
        Retrieve cached decisions or generate new associations.

        Args:
            query: Current candidate context.
            cached_results: Explicitly selected cached streams.

        Returns:
            Validated decisions for the query.

        Raises:
            ValueError: If cache context differs or the model is unavailable.
        """
        if query.task in cached_results:
            result = next(cached_results[query.task], None)

            if result is None or result.query != query:
                raise ValueError(f"Cached candidates differ for {query.alignment_id}")

            validate_result(result)

            return result

        if self._model is None:
            raise ValueError("Uncached alignment requires a language model")

        return align_query(query, self._model, self._mode, self._prompts)

    def align(
        self,
        lemmas: Iterable[Lemma],
        *,
        cached_results: Mapping[AlignmentTask, Iterator[AlignmentResult]] | None = None,
        recorder: Callable[[AlignmentResult], None] | None = None,
    ) -> Iterator[Lemma]:
        """
        Stream aligned copies of collected entries.

        Args:
            lemmas: Original collected entries.
            cached_results: Task-specific decision streams for replay.
            recorder: Optional callback persisting generated decisions.

        Yields:
            Aligned entries with processed lemma-level translations removed.

        Raises:
            ValueError: If cached decisions differ from the collection.
        """
        streams = cached_results or {}

        for lemma in tqdm(lemmas, desc="Aligning senses", unit=" lemma"):
            senses = {
                sense.id: replace(sense, translations=dict(sense.translations))
                for sense in lemma.senses
            }

            translation_tables = lemma.translation_tables

            for task in self._tasks:
                for query in build_queries(lemma, task, self._candidates):
                    try:
                        result = self._evaluate(query, streams)
                    except InvalidModelResponseError as error:
                        with logging_redirect_tqdm(loggers=[getLogger("wsc")]):
                            _LOGGER.warning(
                                "Skipped %s\n\n%s", query.alignment_id, error
                            )

                        continue

                    _LOGGER.debug(
                        "Aligned %s\n\n%s", query.alignment_id, result.response
                    )

                    if recorder is not None:
                        recorder(result)

                    TASK_HANDLERS[task].apply(lemma, senses, query, result.links)

                    if task == AlignmentTask.TRANSLATIONS:
                        translation_tables = ()

            yield replace(
                lemma,
                senses=list(senses.values()),
                translation_tables=translation_tables,
            )

        for task, stream in streams.items():
            if next(stream, None) is not None:
                raise ValueError(f"Unused cached {task} decisions remain")
