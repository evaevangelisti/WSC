"""
What each source is read into, one module apiece.
"""

from .wiktionary import (
    Attestation,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    WordOffset,
)
from .wordnet import Synset

__all__ = [
    "Attestation",
    "Example",
    "Lemma",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "WordOffset",
]
