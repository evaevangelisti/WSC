"""What each source is read into, one module apiece."""

from .wiktionary import (
    Attestation,
    Example,
    Lemma,
    Offset,
    Quotation,
    Sense,
    Sentence,
    TranslationTable,
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
    "TranslationTable",
    "WordNetAlignment",
    "WordNetRelation",
    "WordOffset",
    "WordOffsetSource",
]
