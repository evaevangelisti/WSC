"""
Inputs and associations shared by alignment and research.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, SupportsFloat

from .pos import POS


class AlignmentTask(StrEnum):
    """Resources compared by an alignment."""

    TRANSLATIONS = "translations"
    WORDNET = "wordnet"


class GlossMode(StrEnum):
    """Representations compared by the gloss experiment."""

    LAST = "last"
    FULL = "full"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class AlignmentInstructions:
    """
    Named semantic instructions shared by inference and experiments.

    Attributes:
        name: Profile identifier within its TOML file.
        instruction: General cross-encoder instruction.
        relations: Hypotheses keyed by translation or directed WordNet relation.
    """

    name: str
    instruction: str
    relations: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Definition:
    """
    One candidate in an alignment task.

    Attributes:
        id: Identifier within its side of the task.
        glosses: Definitions ordered from ancestor to leaf.
    """

    id: str
    glosses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlignmentQuery:
    """
    Complete candidate sets for one annotation task.

    Attributes:
        task: Resource being aligned with Wiktionary.
        alignment_id: Stable task identifier.
        lemma_id: Group preventing lemma leakage between experimental splits.
        lemma: Headword providing lexical context.
        pos: Wiktionary part of speech.
        source_definitions: Wiktionary senses.
        target_definitions: Translation groups or WordNet concepts.
    """

    task: AlignmentTask
    alignment_id: str
    lemma_id: str
    lemma: str
    pos: POS
    source_definitions: tuple[Definition, ...]
    target_definitions: tuple[Definition, ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    """
    Semantic hypothesis evaluated against a candidate definition.

    Attributes:
        query: Source definition and requested relation.
        document: Candidate definition.
    """

    query: str
    document: str


class Reranker(Protocol):
    """Text-only cross-encoder prediction contract used by the inference adapter."""

    def predict(
        self,
        inputs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: Literal[True],
    ) -> Iterable[SupportsFloat]:
        """
        Predict one scalar score per text pair.

        Args:
            inputs: Ordered query and document pairs.
            batch_size: Number of pairs evaluated together.
            show_progress_bar: Whether inference displays progress.
            convert_to_numpy: Select array output for single-score predictions.

        Returns:
            Numeric scores convertible to Python floats.
        """
        ...


class Scorer(Protocol):
    """Model boundary allowing offline tests of the complete pipeline."""

    def score(
        self,
        pairs: Sequence[Comparison],
    ) -> list[float]:
        """
        Score hypotheses in input order.

        Args:
            pairs: Hypotheses and candidate definitions.

        Returns:
            One finite, uncalibrated score per pair; larger means stronger support.
        """
        ...


@dataclass(frozen=True, slots=True)
class AlignmentScore:
    """
    Semantic evidence retained before assignment or abstention.

    Attributes:
        source_id: Wiktionary sense identifier.
        target_id: Translation group identifier or WordNet synset identifier.
        relation: Translation association or directed WordNet relation.
        score: Uncalibrated model score.
    """

    source_id: str
    target_id: str
    relation: str
    score: float


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """
    Candidate evidence sufficient to repeat assignment without inference.

    Attributes:
        query: Complete annotation input.
        scores: Scores for every candidate relation.
    """

    query: AlignmentQuery
    scores: tuple[AlignmentScore, ...]
