"""
Where dumps and what is made of them are kept.
"""

from pathlib import Path

from platformdirs import user_cache_dir

LATEST = "latest"

DUMP_NAME = "dump.xml.bz2"
WIKTEXTRACT_NAME = "wiktextract.jsonl.zst"


def _root(
    cache_dir: Path | None,
) -> Path:
    """
    Settle where the cache is, falling back to the usual place.

    Args:
        cache_dir: Where dumps are kept, or None for the usual place.

    Returns:
        The directory to work under.
    """
    return cache_dir or Path(user_cache_dir("wsc"))


def dump_dir(
    cache_dir: Path | None,
    language: str,
    date: str,
) -> Path:
    """
    Name the directory holding one dump and everything derived from it.

    Args:
        cache_dir: Where dumps are kept, or None for the usual place.
        language: Wiktionary's code for the edition.
        date: The day that dump began.

    Returns:
        The directory, whether or not it exists yet.
    """
    return _root(cache_dir) / f"{language}wiktionary-{date}"


def fetched_date(
    cache_dir: Path | None,
    language: str,
    dump_date: str,
) -> str:
    """
    Settle which fetched dump to work on, without asking Wikimedia.

    A dump is known by the directory it sits in, so "latest" is answered from
    the cache alone. That is what lets a machine without a network run all the
    same.

    Args:
        cache_dir: Where dumps are kept, or None for the usual place.
        language: Wiktionary's code for the edition.
        dump_date: The day a dump began, or "latest" for the newest fetched.

    Returns:
        The day the chosen dump began.

    Raises:
        FileNotFoundError: If no such dump has been fetched.
    """
    root = _root(cache_dir)

    if dump_date != LATEST:
        if dump_dir(root, language, dump_date).is_dir():
            return dump_date

        raise FileNotFoundError(
            f"No {language}wiktionary dump of {dump_date} in {root}; fetch it first"
        )

    prefix = f"{language}wiktionary-"
    dates = sorted(
        (
            path.name.removeprefix(prefix)
            for path in root.glob(f"{prefix}*")
            if path.is_dir()
        ),
        reverse=True,
    )

    if not dates:
        raise FileNotFoundError(
            f"No {language}wiktionary dump in {root}; fetch one first"
        )

    return dates[0]
