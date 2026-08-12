"""
Values the program never varies.
"""

# One directory per dump, named after the day it began: the index lists them,
# each reports the state of its jobs, and the archive of pages sits inside.
DUMP_INDEX_URL = "https://dumps.wikimedia.org/{language}wiktionary/"
DUMP_STATUS_URL = (
    "https://dumps.wikimedia.org/{language}wiktionary/{date}/dumpstatus.json"
)
DUMP_URL = "https://dumps.wikimedia.org/{language}wiktionary/{date}/{language}wiktionary-{date}-pages-articles.xml.bz2"

# Wikimedia asks that requests name whoever answers for them.
USER_AGENT = "wsc/{version} (https://github.com/evaevangelisti/WSC)"

# Both shape how a file is downloaded. The read timeout covers a single read.
TIMEOUT = (10, 60)
CHUNK_SIZE = 1024 * 1024

# Written once, read whole on every collection.
COMPRESSION_LEVEL = 10
