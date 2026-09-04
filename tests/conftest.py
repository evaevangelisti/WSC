"""
Fixtures the whole suite shares.

A feature is exercised from outside, over what the generators draw, and the
test tree mirrors the package.
"""

import gzip
import json
from collections.abc import Callable, Iterable
from compression import zstd
from itertools import count
from pathlib import Path

import pytest
from documents import WORDNET, lexicon
from engines import WhitespaceEngine
from hypothesis import HealthCheck, settings
from kwic import Locator
from strategies import RawJson

from wsc.upstream import cache

# A property reaching the disk is timed by the machine it runs on, and the
# fixtures below hand out a directory per call rather than per test.
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
        A builder handing back an empty directory, so that a property drawing
        a hundred examples writes each of them somewhere else.
    """
    directories = count()

    def build() -> Path:
        path = tmp_path / f"{next(directories):03d}"
        path.mkdir()

        return path

    return build


@pytest.fixture
def write_entries() -> Callable[[Path, Iterable[RawJson]], Path]:
    """
    Write raw entries where a reader expects to find them.

    Returns:
        A writer picking its compression off the suffix of the path, so that
        a test names the format it means by naming the file.
    """

    def write(
        path: Path,
        entries: Iterable[RawJson],
    ) -> Path:
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
        A builder placing a dump where the fetch command would have.
    """

    def build(
        cache_dir: Path,
        language: str = "en",
        date: str = "20260801",
    ) -> Path:
        path = cache.dump_dir(cache_dir, language, date) / cache.DUMP_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"a dump")

        return path

    return build


@pytest.fixture
def fetch_wordnet() -> Callable[..., Path]:
    """
    Stand in for a finished fetch of the wordnet, without the network.

    Returns:
        A builder placing the wordnet where the wordnet command would have.
        It holds a whole one, gzipped as the release publishes it, since the
        command goes on to read what it fetched.
    """

    def build(
        cache_dir: Path,
        version: str = "2025",
        elements: Iterable[str] = WORDNET,
    ) -> Path:
        path = cache.wordnet_dir(cache_dir, version) / cache.WORDNET_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(gzip.compress(lexicon(*elements).encode()))

        return path

    return build


@pytest.fixture
def read_wordnet(
    fetch_wordnet: Callable[..., Path],
) -> Callable[..., Path]:
    """
    Stand in for a finished fetch and read of the wordnet.

    Args:
        fetch_wordnet: Places the wordnet the read would have read.

    Returns:
        A builder placing the synsets where the wordnet command would have.
    """

    def build(
        cache_dir: Path,
        version: str = "2025",
        lines: Iterable[str] = (),
    ) -> Path:
        _ = fetch_wordnet(cache_dir, version)

        path = cache.wordnet_dir(cache_dir, version) / cache.SYNSETS_NAME
        _ = path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

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
        language: str = "en",
        date: str = "20260801",
    ) -> Path:
        _ = fetch_dump(cache_dir, language, date)

        path = cache.dump_dir(cache_dir, language, date) / cache.WIKTEXTRACT_NAME

        return write_entries(path, entries)

    return build


@pytest.fixture(scope="session")
def locator() -> Locator:
    """
    Hand out the search every extraction reads with.

    Returns:
        A search over an engine that never varies and loads nothing.
    """
    return Locator(WhitespaceEngine())
