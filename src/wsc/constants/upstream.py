"""
Source locations and download defaults.
"""

DUMP_INDEX_URL = "https://dumps.wikimedia.org/enwiktionary/"
"""Where the edition lists its dumps, one directory per day one began."""

DUMP_STATUS_URL = "https://dumps.wikimedia.org/enwiktionary/{date}/dumpstatus.json"
"""Where one dump reports its jobs, and so whether it finished."""

DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/{date}/enwiktionary-{date}-pages-articles.xml.bz2"
"""The archive of pages sitting inside a dump, which is the dump itself."""

KAIKKI_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
"""Preparsed Wiktionary entries that avoid running wiktextract locally."""

WORDNET_INDEX_URL = "https://en-word.net/downloads"
"""Where WordNet lists its editions, one file per year and format."""

WORDNET_URL = "https://en-word.net/downloads/english-wordnet-{version}.xml.gz"
"""Where one edition is published, in WN-LMF."""

USER_AGENT = "wsc/{version} (https://github.com/evaevangelisti/WSC)"
"""How a request names itself, Wikimedia asking that it name someone."""

TIMEOUT = (10, 60)
"""Seconds to wait for the connection, then for a read; not for the download."""

CHUNK_SIZE = 1024 * 1024
"""How much of a download is held in memory before it reaches the disk."""
