"""
Domain model for Wiktionary entries parsed by wiktextract.
"""

from dataclasses import dataclass, field

from ..pos import POS

type WordOffset = tuple[int, int]
"""Half-open range of code points, as Python slices them."""

type Translations = dict[str, dict[str, frozenset[str]]]
"""What other languages call an entry: the gloss a translation table heads,
then the language, then the words it offers."""


@dataclass(frozen=True, slots=True)
class Attestation:
    """
    Evidence that a sense is in use.

    Attributes:
        text: The sentence.
        word_offsets: Where the lemma occurs in it, leftmost first.
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
        synonyms: Other words standing for this meaning alone.
        tags: Labels of grammar and register, such as transitive or obsolete.
        topics: Subject fields the sense belongs to, such as mathematics.
        sentences: The examples and quotations attached to this sense.
        sense_ids: What Wiktionary names the sense, such as en:Q23622.
        wikidata_ids: The Wikidata items it was tied to, such as Q23622.
    """

    id: str
    glosses: tuple[str, ...]
    synonyms: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    sentences: list[Sentence] = field(default_factory=list)
    sense_ids: tuple[str, ...] = ()
    wikidata_ids: tuple[str, ...] = ()

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
        id: Identifies the entry, such as bank.noun.7d20e4c8.
        lemma: The headword.
        pos: Its part of speech.
        variants: How else the lemma is written, from the entries stating
        themselves to be a form of it and from its own Alternative forms.
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
