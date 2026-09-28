"""
Expose markup cleanup and reference removal.
"""

from .formatting import (
    is_literal_markup,
    is_unrecoverable,
    normalize_formatting,
    normalize_statement,
)
from .references import (
    BIBLIOGRAPHY,
    METADATA,
    NAVIGATION,
    clean_definition_references,
    remove_references,
)

__all__ = [
    "BIBLIOGRAPHY",
    "METADATA",
    "NAVIGATION",
    "clean_definition_references",
    "is_literal_markup",
    "is_unrecoverable",
    "normalize_formatting",
    "normalize_statement",
    "remove_references",
]
