"""
Reading a Wiktionary dump in the markup it was written in.

What one page says about another is read here.
"""

from .extractor import DumpExtractor
from .translations import PageTranslations

__all__ = [
    "DumpExtractor",
    "PageTranslations",
]
