"""Expose alignment task handlers and query construction."""

from collections.abc import Iterator

from ...models import Lemma
from ...models.alignment import AlignmentQuery, AlignmentTask
from ..candidates import SynsetCandidates
from .base import AlignmentHandler, build_definitions
from .synsets import SynsetHandler
from .translations import TranslationHandler

__all__ = [
    "TASK_HANDLERS",
    "AlignmentHandler",
    "SynsetHandler",
    "TranslationHandler",
    "build_definitions",
    "build_queries",
]

TASK_HANDLERS: dict[AlignmentTask, AlignmentHandler] = {
    AlignmentTask.TRANSLATIONS: TranslationHandler(),
    AlignmentTask.SYNSETS: SynsetHandler(),
}
"""Register query construction, relation constraints, and result application."""


def build_queries(
    lemma: Lemma,
    task: AlignmentTask,
    candidates: SynsetCandidates,
) -> Iterator[AlignmentQuery]:
    """
    Expose complete candidate sets without semantic filtering.

    Args:
        lemma: Collected entry.
        task: Resource to align.
        candidates: Synset candidate index.

    Yields:
        One query containing all source senses for the requested resource.
    """
    yield from TASK_HANDLERS[task].queries(lemma, candidates)
