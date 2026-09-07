"""
Domain model for Wiktionary entries parsed by wiktextract.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from ..pos import POS
from .wordnet import WordNetAlignment

type Offset = tuple[int, int]
"""Half-open code-point range."""


class WordOffsetSource(StrEnum):
    """
    Method proposing a word offset.

    Attributes:
        BOLD: Range supplied by Wiktextract's bold text.
        LEMMATIZER: Range supplied by the lemmatizer pipeline.
    """

    BOLD = "bold"
    LEMMATIZER = "lemmatizer"


@dataclass(frozen=True, slots=True)
class WordOffset:
    """
    Candidate range and its supporting methods.

    Attributes:
        offset: Half-open code-point range.
        sources: Methods supporting the candidate.
    """

    offset: Offset
    sources: tuple[WordOffsetSource, ...]


@dataclass(frozen=True, slots=True)
class Attestation:
    """
    Evidence that a sense is in use.

    Attributes:
        text: The sentence.
        word_offsets: Candidate ranges and the methods supporting them.
    """

    text: str
    word_offsets: tuple[WordOffset, ...] = field(default=(), kw_only=True)


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
        id: Identifies the sense, such as bank.noun.3f9c1a2b.
        glosses: The gloss chain, outermost first. Never empty.
        etymology: Which etymology of the entry it belongs to, empty where
        the page states only one.
        synonyms: Other words standing for this meaning alone.
        topics: Subject fields the sense belongs to, such as mathematics.
        tags: Labels of grammar and register, such as transitive or obsolete.
        sentences: The examples and quotations attached to this sense.
        translations: Words for this sense, grouped by language.
        wikidata_ids: The Wikidata items it was tied to, such as Q23622.
        wordnet: WordNet concepts and their semantic relations.
    """

    id: str
    glosses: tuple[str, ...]
    etymology: str = ""
    synonyms: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    sentences: list[Sentence] = field(default_factory=list)
    translations: dict[str, frozenset[str]] = field(default_factory=dict, kw_only=True)
    wikidata_ids: tuple[str, ...] = ()
    wordnet: tuple[WordNetAlignment, ...] = field(default=(), kw_only=True)

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


type Translations = dict[str, dict[str, frozenset[str]]]
"""What other languages call an entry: the gloss a translation table heads,
then the language, then the words it offers."""


@dataclass(slots=True)
class Lemma:
    """
    A written form with a part of speech.

    Wiktionary splits an entry by etymology, which the senses carry
    instead.

    Attributes:
        id: Identifies the entry, such as bank.noun.
        lemma: The headword.
        pos: Its part of speech.
        variants: Lemma identifiers of alternative spellings, such as colour.noun.
        senses: Its meanings, in the order Wiktionary lists them.
        translations: What other languages call it, hung off the entry rather
        than off a sense, the way Wiktionary writes them.
    """

    id: str
    lemma: str
    pos: POS
    variants: frozenset[str] = frozenset()
    senses: list[Sense] = field(default_factory=list)
    translations: Translations = field(default_factory=dict)
