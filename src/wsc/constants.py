"""
Values the program never varies.
"""

DUMP_INDEX_URL = "https://dumps.wikimedia.org/{language}wiktionary/"
"""Where an edition lists its dumps, one directory per day one began."""

DUMP_STATUS_URL = (
    "https://dumps.wikimedia.org/{language}wiktionary/{date}/dumpstatus.json"
)
"""Where one dump reports its jobs, and so whether it finished."""

DUMP_URL = "https://dumps.wikimedia.org/{language}wiktionary/{date}/{language}wiktionary-{date}-pages-articles.xml.bz2"
"""The archive of pages sitting inside a dump, which is the dump itself."""

WORDNET_INDEX_URL = "https://en-word.net/downloads"
"""Where the wordnet lists its editions, one file per year and format."""

WORDNET_URL = "https://en-word.net/downloads/english-wordnet-{version}.xml.gz"
"""Where one edition is published, in WN-LMF."""

USER_AGENT = "wsc/{version} (https://github.com/evaevangelisti/WSC)"
"""How a request names itself, Wikimedia asking that it name someone."""

TIMEOUT = (10, 60)
"""Seconds to wait for the connection, then for a read; not for the download."""

CHUNK_SIZE = 1024 * 1024
"""How much of a download is held in memory before it reaches the disk."""

COMPRESSION_LEVEL = 10
"""How hard a parsed dump is packed, written once and read whole every time."""
