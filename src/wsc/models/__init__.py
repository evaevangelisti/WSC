"""
Data structures passed around the pipeline.
"""

from .engine import Engine
from .pos import POS
from .wiktionary import (
    Attestation,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
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
    "Translations",
    "WordOffset",
]
