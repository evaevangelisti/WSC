"""
Where a Wiktionary dump comes from, where it is kept, and what is made of it.

The repository names what Wikimedia has published, the download brings one
across, the cache holds on to it, and wiktextract turns its markup into
entries something can read.
"""

from . import cache, repository, wiktextract
from .download import download

__all__ = [
    "cache",
    "download",
    "repository",
    "wiktextract",
]
