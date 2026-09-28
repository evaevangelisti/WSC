"""
Expose Wiktionary sentence parsing and references.
"""

from .parsing import clean_sentence, parse_sentences
from .references import clean_reference, parse_year, read_source

__all__ = [
    "clean_reference",
    "clean_sentence",
    "parse_sentences",
    "parse_year",
    "read_source",
]
