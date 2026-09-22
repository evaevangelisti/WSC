"""
Fixtures the whole suite shares.

A feature is exercised from outside, over what the generators draw, and the test tree
mirrors the package.
"""

import bz2
import gzip
import json
from collections.abc import Callable, Iterable
from compression import zstd
from itertools import count
from pathlib import Path

import pytest
from documents import dump
from engines import WhitespaceEngine
from hypothesis import HealthCheck, settings
from kwic import Locator
from strategies import RawJson

from wsc.extract import write_off_page_translations
from wsc.upstream import cache

# Filesystem properties use isolated directories for each generated example.
settings.register_profile(
    "wsc",
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("wsc")


@pytest.fixture
def workspace(
    tmp_path: Path,
) -> Callable[[], Path]:
    """
    Set aside a directory of one's own, as often as one is asked for.

    Args:
        tmp_path: The directory pytest set aside for this test.

    Returns:
        A builder creating an isolated directory for each generated example.
    """
    directories = count()

    def build() -> Path:
        """
        Create a fresh directory for one generated example.

        Returns:
            The newly created workspace path.
        """
        path = tmp_path / f"{next(directories):03d}"
        path.mkdir()

        return path

    return build


@pytest.fixture
def write_entries() -> Callable[[Path, Iterable[RawJson]], Path]:
    """
    Write raw entries where a reader expects to find them.

    Returns:
        A writer selecting compression from the file suffix.
    """

    def write(
        path: Path,
        entries: Iterable[RawJson],
    ) -> Path:
        """
        Write entries using the compression selected by the path.

        Args:
            path: Destination path selecting the compression format.
            entries: Source entries to serialize and extract.

        Returns:
            The populated source path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = "".join(
            f"{json.dumps(entry, ensure_ascii=False)}\n" for entry in entries
        )

        match path.suffix:
            case ".zst":
                _ = path.write_bytes(zstd.compress(lines.encode()))

            case ".gz":
                _ = path.write_bytes(gzip.compress(lines.encode()))

            case _:
                _ = path.write_text(lines, encoding="utf-8")

        return path

    return write


@pytest.fixture
def fetch_dump() -> Callable[..., Path]:
    """
    Stand in for a finished fetch, without the network.

    Returns:
        A builder placing a complete dump in the fetch cache.
    """

    def build(
        cache_dir: Path,
        date: str = "20260801",
        pages: Iterable[str] = (),
    ) -> Path:
        """
        Place a compressed Wiktionary dump in the test cache.

        Args:
            cache_dir: Isolated source cache directory.
            date: Dump date used to name the cache directory.
            pages: Page elements included in the generated dump.

        Returns:
            The cached dump path.
        """
        path = cache.dump_dir(cache_dir, date) / cache.DUMP_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(bz2.compress(dump(*pages).encode()))

        return path

    return build


@pytest.fixture
def parse_dump(
    fetch_dump: Callable[..., Path],
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
) -> Callable[..., Path]:
    """
    Stand in for a finished fetch and parse, without wiktextract.

    Args:
        fetch_dump: Places the dump the parse would have read.
        write_entries: Writes the entries the parse would have produced.

    Returns:
        A builder placing wiktextract output where the parse command would.
    """

    def build(
        cache_dir: Path,
        entries: Iterable[RawJson],
        date: str = "20260801",
    ) -> Path:
        """
        Populate the parsed cache and supplemental translations.

        Args:
            cache_dir: Isolated source cache directory.
            entries: Source entries to serialize and extract.
            date: Dump date used to name the cache directory.

        Returns:
            The cached extraction path.
        """
        _ = fetch_dump(cache_dir, date)

        dump_dir = cache.dump_dir(cache_dir, date)
        write_off_page_translations(dump_dir / cache.OFF_PAGE_TRANSLATIONS_NAME, {})

        return write_entries(dump_dir / cache.WIKTEXTRACT_NAME, entries)

    return build


@pytest.fixture(scope="session")
def locator() -> Locator:
    """
    Hand out the search every extraction reads with.

    Returns:
        A search over an engine that never varies and loads nothing.
    """
    return Locator(WhitespaceEngine())
