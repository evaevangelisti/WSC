"""
Where a source comes from, where it is kept, and what is made of it.

The repositories name what has been published, the download brings one file
across, the cache holds on to it, and wiktextract turns a dump's markup into
entries something can read.
"""

from . import cache, repositories, wiktextract
from .download import download

__all__ = [
    "cache",
    "download",
    "repositories",
    "wiktextract",
]
