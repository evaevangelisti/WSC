"""
What each source is read into, one module apiece.
"""

from .synsets import Synset, SynsetAlignment, SynsetRelation, SynsetResource
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

__all__ = [
    "Attestation",
    "Example",
    "Lemma",
    "Offset",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "SynsetAlignment",
    "SynsetRelation",
    "SynsetResource",
    "TranslationTable",
    "WordOffset",
    "WordOffsetSource",
]
