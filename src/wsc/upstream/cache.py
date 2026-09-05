"""
Where the sources and what is made of them are kept.
"""

from pathlib import Path

from platformdirs import user_cache_dir

LATEST = "latest"

DUMP_NAME = "dump.xml.bz2"
ARCHIVE_NAME = "archive.jsonl.gz"
WIKTEXTRACT_NAME = "wiktextract.jsonl.zst"
OFF_PAGE_TRANSLATIONS_NAME = "off-page-translations.json"
WORDNET_NAME = "wordnet.xml.gz"
SYNSETS_NAME = "synsets.jsonl"

_WIKTIONARY = "wiktionary"
_WORDNET = "wordnet"


def _root(
    cache_dir: Path | None,
) -> Path:
    """
    Settle where the cache is, falling back to the usual place.

    Args:
        cache_dir: Where the sources are kept, or None for the usual place.

    Returns:
        The directory to work under.
    """
    return cache_dir or Path(user_cache_dir("wsc"))


def _edition_dir(
    cache_dir: Path | None,
) -> Path:
    """
    Name the directory holding every dump of the Wiktionary edition.

    Args:
        cache_dir: Where the sources are kept, or None for the usual place.

    Returns:
        The directory, whether or not it exists yet.
    """
    return _root(cache_dir) / _WIKTIONARY


def dump_dir(
    cache_dir: Path | None,
    date: str,
) -> Path:
    """
    Name the directory holding one dump and everything derived from it.

    Args:
        cache_dir: Where the sources are kept, or None for the usual place.
        date: The day that dump began.

    Returns:
        The directory, whether or not it exists yet.
    """
    return _edition_dir(cache_dir) / date


def fetched_date(
    cache_dir: Path | None,
    dump_date: str,
) -> str:
    """
    Settle which fetched dump to work on, without asking Wikimedia.

    A dump is known by the directory it sits in, so "latest" is answered from
    the cache alone, and a machine without a network runs all the same.

    Args:
        cache_dir: Where the sources are kept, or None for the usual place.
        dump_date: The day a dump began, or "latest" for the newest fetched.

    Returns:
        The day the chosen dump began.

    Raises:
        FileNotFoundError: If no such dump has been fetched.
    """
    edition_dir = _edition_dir(cache_dir)

    if dump_date != LATEST:
        if (edition_dir / dump_date).is_dir():
            return dump_date

        raise FileNotFoundError(
            f"No dump of {dump_date} in {edition_dir}; fetch it first"
        )

    dates = sorted(
        (path.name for path in edition_dir.glob("*") if path.is_dir()),
        reverse=True,
    )

    if not dates:
        raise FileNotFoundError(f"No dump in {edition_dir}; fetch one first")

    return dates[0]


def wordnet_dir(
    cache_dir: Path | None,
    version: str,
) -> Path:
    """
    Name the directory holding one wordnet and everything derived from it.

    Args:
        cache_dir: Where the sources are kept, or None for the usual place.
        version: The edition of the wordnet, as 2025.

    Returns:
        The directory, whether or not it exists yet.
    """
    return _root(cache_dir) / _WORDNET / version
