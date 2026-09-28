"""
Where the sources and what is made of them are kept.
"""

from pathlib import Path

from platformdirs import user_cache_dir

from ..files import COMPRESSED_SUFFIXES

LATEST = "latest"

DUMP_NAME = "dump.xml.bz2"

ARCHIVE_NAME = "archive.jsonl.gz"
WIKTEXTRACT_NAME = "wiktextract.jsonl.zst"

OFF_PAGE_TRANSLATIONS_NAME = "off-page-translations.json"
WIKIDATA_IDS_NAME = "wikidata-ids.json"
MARKUP_INDEX_NAME = "markup-index.json"
RESOURCE_DIRECTORY = "resources"

_WIKTIONARY = "wiktionary"


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
        The requested cache directory path.
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
        The requested cache directory path.
    """
    return _edition_dir(cache_dir) / date


def resource_dir(
    dump_directory: Path,
) -> Path:
    """
    Locate supplemental resources for one dump.

    Args:
        dump_directory: Directory holding the dump.

    Returns:
        Directory for derived resources.
    """
    return dump_directory / RESOURCE_DIRECTORY


def migrate_resources(
    dump_directory: Path,
) -> Path:
    """
    Move existing supplemental resources into their shared directory.

    Args:
        dump_directory: Directory holding the dump and legacy resources.

    Returns:
        Directory for derived resources.
    """
    directory = resource_dir(dump_directory)

    for name in (
        OFF_PAGE_TRANSLATIONS_NAME,
        WIKIDATA_IDS_NAME,
        MARKUP_INDEX_NAME,
    ):
        for suffix in ("", *COMPRESSED_SUFFIXES):
            source = dump_directory / f"{name}{suffix}"
            destination = directory / source.name

            if source.is_file() and not destination.exists():
                directory.mkdir(parents=True, exist_ok=True)
                _ = source.replace(destination)

    return directory


def fetched_date(
    cache_dir: Path | None,
    dump_date: str,
) -> str:
    """
    Settle which fetched dump to work on, without asking Wikimedia.

    A dump is known by the directory it sits in, so "latest" is answered from the cache
    alone, and a machine without a network runs all the same.

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


def alignment_dir(
    cache_dir: Path | None,
) -> Path:
    """
    Locate reusable alignment evidence.

    Args:
        cache_dir: Source cache directory or platform default.

    Returns:
        Directory holding separate resource TSV files.
    """
    return _root(cache_dir) / "alignment"
