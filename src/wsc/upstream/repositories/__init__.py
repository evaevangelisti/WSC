"""
Where each source is published, and under what address.

One module apiece, since Wikimedia and the wordnet each list what they hold
in their own way.
"""

from . import wiktionary, wordnet

__all__ = [
    "wiktionary",
    "wordnet",
]
