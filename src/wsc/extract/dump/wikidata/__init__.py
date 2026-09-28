"""
Expose Wikidata indexing and cache operations.
"""

from .association import index_wikidata_ids
from .cache import WikidataIds, read_wikidata_ids, write_wikidata_ids
from .pages import DumpDefinition, read_page_wikidata

__all__ = [
    "DumpDefinition",
    "WikidataIds",
    "index_wikidata_ids",
    "read_page_wikidata",
    "read_wikidata_ids",
    "write_wikidata_ids",
]
