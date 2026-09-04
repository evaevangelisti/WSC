"""
Data structures passed around the pipeline.
"""

from .engine import Engine
from .pos import POS
from .resources import (
    Attestation,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    Synset,
    Translations,
    WordOffset,
)

__all__ = [
    "POS",
    "Attestation",
    "Engine",
    "Example",
    "Lemma",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "Translations",
    "WordOffset",
]
