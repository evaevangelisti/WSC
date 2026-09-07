"""Define inputs, decisions, and model contracts for lexical alignment."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .pos import POS


class AlignmentTask(StrEnum):
    """Identify the resource aligned with Wiktionary."""

    TRANSLATIONS = "translations"
    WORDNET = "wordnet"


class GlossMode(StrEnum):
    """Select the Wiktionary definition representation."""

    LAST = "last"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """
    Configure language model inference.

    Attributes:
        model: Model identifier exposed by the server.
        temperature: Sampling temperature.
        maximum_tokens: Maximum number of generated tokens.
        url: OpenAI-compatible server endpoint.
        reasoning_effort: Optional reasoning setting supported by the server.
    """

    model: str
    temperature: float = 0.0
    maximum_tokens: int = 4096
    url: str = "http://localhost:8000/v1"
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class AlignmentPrompts:
    """
    Store one prompt template for each alignment task.

    Attributes:
        name: Prompt file label.
        tasks: Prompt templates indexed by alignment task.
    """

    name: str
    tasks: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Definition:
    """
    Represent a lexical sense and its context.

    Attributes:
        id: Identifier within its side of the task.
        glosses: Definitions ordered from ancestor to leaf.
        synonyms: Lexical forms expressing this sense.
    """

    id: str
    glosses: tuple[str, ...]
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlignmentQuery:
    """
    Store the complete candidate sets for an alignment.

    Attributes:
        task: Resource being aligned with Wiktionary.
        alignment_id: Stable task identifier.
        lemma_id: Entry identifier used for experimental grouping.
        lemma: Headword providing lexical context.
        pos: Wiktionary part of speech.
        source_definitions: Wiktionary senses.
        target_definitions: Candidate definitions.
    """

    task: AlignmentTask
    alignment_id: str
    lemma_id: str
    lemma: str
    pos: POS
    source_definitions: tuple[Definition, ...]
    target_definitions: tuple[Definition, ...]


@dataclass(frozen=True, slots=True)
class AlignmentLink:
    """
    Identify an accepted semantic association.

    Attributes:
        source_id: Wiktionary sense identifier.
        target_id: Candidate identifier.
        relation: Semantic relation directed from Wiktionary to the candidate.
        reason: Brief evidence supporting the association.
    """

    source_id: str
    target_id: str
    relation: str
    reason: str


@dataclass(frozen=True, slots=True)
class AlignmentDecision:
    """
    Record a source decision and its accepted associations.

    Attributes:
        source_id: Wiktionary sense identifier.
        links: Accepted links for this source.
    """

    source_id: str
    links: tuple[AlignmentLink, ...] = ()


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """
    Retain decisions and the original model response.

    Attributes:
        query: Complete model input.
        decisions: One decision per source sense.
        response: Original generated JSON retained for review.
    """

    query: AlignmentQuery
    decisions: tuple[AlignmentDecision, ...]
    response: str = ""

    @property
    def links(
        self,
    ) -> tuple[AlignmentLink, ...]:
        """Return accepted associations in source order."""
        return tuple(link for decision in self.decisions for link in decision.links)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """
    Provide the prompt and its structured response schema.

    Attributes:
        prompt: Complete lexical alignment instructions and definitions.
        schema: JSON schema accepted by structured-output providers.
    """

    prompt: str
    schema: dict[str, object]


class LanguageModel(Protocol):
    """Generate structured alignment decisions."""

    def generate(
        self,
        request: ModelRequest,
    ) -> str:
        """
        Generate a response for one alignment request.

        Args:
            request: Prompt and response schema.

        Returns:
            Generated JSON text.
        """
        ...
