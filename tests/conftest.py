"""
Fixtures the whole suite shares.

The test tree mirrors the package: tests/extract/test_wiktionary.py covers
src/wsc/extract/wiktionary.py, and a module at the top of the tree covers one
of the same name in src/wsc. A new source or format is a new file in the
matching directory, never an addition to an existing one.

What every directory reads lives here; what one directory alone needs belongs
in a conftest.py of its own, next to the tests that ask for it.

A test is documented in a single line, its parameters being fixtures rather
than arguments. A fixture is documented like any other function.
"""

import gzip
import json
from collections.abc import Callable, Iterable
from compression import zstd
from pathlib import Path

import pytest

from wsc import cache

type RawJson = dict[str, object]
"""One decoded JSON object, as wiktextract writes them."""


# The builders below spell out only what a test cares about.


@pytest.fixture
def make_example() -> Callable[..., RawJson]:
    """
    Build one raw example, quoted from a source or not.

    Returns:
        A builder taking the sentence and, optionally, its reference.
    """

    def build(
        text: str = "A sentence.",
        ref: str | None = None,
    ) -> RawJson:
        raw: RawJson = {"text": text}
        if ref is not None:
            raw["ref"] = ref

        return raw

    return build


@pytest.fixture
def make_sense() -> Callable[..., RawJson]:
    """
    Build one raw sense.

    Returns:
        A builder taking the gloss chain, its labels and its examples.
    """

    def build(
        glosses: list[str] | None = None,
        tags: list[str] | None = None,
        topics: list[str] | None = None,
        examples: list[RawJson] | None = None,
    ) -> RawJson:
        return {
            "glosses": ["A meaning."] if glosses is None else glosses,
            "tags": tags or [],
            "topics": topics or [],
            "examples": examples or [],
        }

    return build


@pytest.fixture
def make_entry(
    make_sense: Callable[..., RawJson],
) -> Callable[..., RawJson]:
    """
    Build one raw entry, with a single plain sense unless told otherwise.

    Args:
        make_sense: Builds the sense an entry falls back on.

    Returns:
        A builder taking the headword, its part of speech and its senses.
    """

    def build(
        word: str = "bank",
        pos: str = "noun",
        lang_code: str = "en",
        senses: list[RawJson] | None = None,
    ) -> RawJson:
        return {
            "word": word,
            "pos": pos,
            "lang_code": lang_code,
            "senses": [make_sense()] if senses is None else senses,
        }

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
def cache_dir(
    tmp_path: Path,
) -> Path:
    """
    Give each test a cache of its own, so none inherits another's dumps.

    Args:
        tmp_path: The directory pytest set aside for this test.

    Returns:
        The directory to pass wherever a cache is asked for.
    """
    return tmp_path / "cache"


@pytest.fixture
def fetch_dump(
    cache_dir: Path,
) -> Callable[..., Path]:
    """
    Stand in for a finished fetch, without the network.

    Args:
        cache_dir: Where dumps are kept.

    Returns:
        A builder placing a dump where the fetch command would have.
    """

    def build(
        language: str = "en",
        date: str = "20260801",
    ) -> Path:
        path = cache.dump_dir(cache_dir, language, date) / cache.DUMP_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"a dump")

        return path

    return build


@pytest.fixture
def parse_dump(
    cache_dir: Path,
    fetch_dump: Callable[..., Path],
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
) -> Callable[..., Path]:
    """
    Stand in for a finished fetch and parse, without wiktextract.

    Args:
        cache_dir: Where dumps are kept.
        fetch_dump: Places the dump the parse would have read.
        write_entries: Writes the entries the parse would have produced.

    Returns:
        A builder placing wiktextract output where the parse command would.
    """

    def build(
        entries: Iterable[RawJson],
        language: str = "en",
        date: str = "20260801",
    ) -> Path:
        _ = fetch_dump(language, date)

        path = cache.dump_dir(cache_dir, language, date) / cache.WIKTEXTRACT_NAME

        return write_entries(path, entries)

    return build
