"""
Domain model for Wiktionary entries parsed by wiktextract.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class POS(StrEnum):
    """
    Part-of-speech tags the collector keeps.

    Values are wiktextract's own codes, so POS("adj") converts directly.
    """

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"


@dataclass(frozen=True, slots=True)
class Attestation:
    """
    Evidence that a sense is in use.

    Attributes:
        text: The sentence.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Example(Attestation):
    """
    A usage example written by a Wiktionary editor.
    """


@dataclass(frozen=True, slots=True)
class Quotation(Attestation):
    """
    A sentence cited from a real source, which the reference names.

    Attributes:
        reference: The source, as Wiktionary formats it.
        year: Year read off the reference, or None if it names none.
    """

    reference: str
    year: int | None = None


type Sentence = Example | Quotation
"""Either kind of attestation."""


@dataclass(slots=True)
class Sense:
    """
    One meaning of a lemma, with the sentences that illustrate it.

    Attributes:
        id: Identifies the sense, such as bank.noun.2.03.
        glosses: The gloss chain, outermost first. Never empty.
        tags: Labels of grammar and register, such as transitive or obsolete.
        topics: Subject fields the sense belongs to, such as mathematics.
        sentences: The examples and quotations attached to this sense.
    """

    id: str
    glosses: tuple[str, ...]
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def gloss(
        self,
    ) -> str:
        """The sense's own gloss, without its parents."""
        return self.glosses[-1]

    @property
    def definition(
        self,
    ) -> str:
        """The gloss chain joined into a self-contained definition."""
        return " ".join(self.glosses)

    @property
    def depth(
        self,
    ) -> int:
        """Nesting level: 1 for a top-level sense, 2 for a sub-sense."""
        return len(self.glosses)


@dataclass(slots=True)
class Lemma:
    """
    A written form with a part of speech.

    Attributes:
        id: Identifies the entry, such as bank.noun.2.
        lemma: The headword.
        pos: Its part of speech.
        senses: Its meanings, in the order Wiktionary lists them.
    """

    id: str
    lemma: str
    pos: POS
    senses: list[Sense] = field(default_factory=list)
