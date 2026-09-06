"""
Data structures passed around the pipeline.
"""

from .engine import Engine
from .pos import POS
from .resources import (
    Attestation,
    Example,
    Lemma,
    Offset,
    Quotation,
    Sense,
    Sentence,
    Synset,
    Translations,
    WordOffset,
    WordOffsetSource,
)

__all__ = [
    "POS",
    "Attestation",
    "Engine",
    "Example",
    "Lemma",
    "Offset",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "Translations",
    "WordOffset",
    "WordOffsetSource",
]
