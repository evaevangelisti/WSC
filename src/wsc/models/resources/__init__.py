"""What each source is read into, one module apiece."""

from .wiktionary import (
    Attestation,
    Example,
    Lemma,
    Offset,
    Quotation,
    Sense,
    Sentence,
    Translations,
    WordOffset,
    WordOffsetSource,
)
from .wordnet import Synset, WordNetAlignment, WordNetRelation

__all__ = [
    "Attestation",
    "Example",
    "Lemma",
    "Offset",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "Translations",
    "WordNetAlignment",
    "WordNetRelation",
    "WordOffset",
    "WordOffsetSource",
]
