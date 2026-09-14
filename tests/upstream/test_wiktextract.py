"""
Tests for src/wsc/upstream/wiktextract.py.

Wiktextract itself is never run: what is tested is the plumbing around it, a stand-in
subprocess keeping real streams and real exit codes.
"""

import json
import string
import subprocess
import sys
from collections.abc import Callable, Iterable
from compression import zstd
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wsc.constants import LANGUAGE
from wsc.upstream import wiktextract

_ENTRY_LINES = st.dictionaries(
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
    st.text(alphabet=string.ascii_letters, max_size=5),
    max_size=3,
).map(
    json.dumps,
)

_REPORT_LINES = st.text(
    alphabet=f"{string.digits}{string.ascii_letters}{string.punctuation} ",
    max_size=20,
).filter(
    lambda line: not line.startswith("{"),
)

_SPAWNS = settings(max_examples=10)


def _preserve_entry(
    entry: object,
) -> object:
    """
    Keep an entry as it stands, cutting nothing down.

    What to cut is the schema's business, tested elsewhere.

    Args:
        entry: One entry, as the extraction wrote it.

    Returns:
        The same entry.
    """
    return entry


class _SilentProcess:
    """
    A process handing back no stream to read, as typeshed allows one to.

    Attributes:
        stdout: Nothing, which is what the parse guards against.
    """

    stdout: IO[bytes] | None = None

    def __enter__(
        self,
    ) -> Self:
        """
        Stand in for a process entered as a context manager.

        Returns:
            The process itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stand in for a process left, which has nothing to release."""


@pytest.fixture
def stub_wiktextract(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[list[str]]]:
    """
    Answer in wiktextract's place, down the stream it would have written to.

    Args:
        monkeypatch: Puts the stand-in in place, and takes it away after.

    Returns:
        A builder recording commands with configured output and exit codes.
    """
    real_popen = subprocess.Popen

    def build(
        lines: Iterable[str],
        return_code: int = 0,
    ) -> list[list[str]]:
        commands: list[list[str]] = []

        script = "".join(f"print({line!r})\n" for line in lines)
        script += f"raise SystemExit({return_code})"

        def popen(
            command: list[str],
            stdout: int | None = None,
        ) -> subprocess.Popen[bytes]:
            commands.append(command)

            return real_popen([sys.executable, "-c", script], stdout=stdout)

        monkeypatch.setattr(subprocess, "Popen", popen)

        return commands

    return build


@pytest.fixture
def dump_path(
    tmp_path: Path,
) -> Path:
    """
    A dump for wiktextract to be pointed at.

    Args:
        tmp_path: The directory pytest set aside for this test.

    Returns:
        The path of a file standing in for a Wiktionary dump.
    """
    path = tmp_path / "dump.xml.bz2"
    _ = path.write_bytes(b"a dump")

    return path


def read(
    path: Path,
) -> list[str]:
    """
    Read back the entries a parse wrote.

    Args:
        path: The compressed JSONL the parse produced.

    Returns:
        One line per entry, without its newline.
    """
    with zstd.open(path, "rt", encoding="utf-8") as file:
        return file.read().splitlines()


class TestParse:
    """Turning a dump into the compressed JSONL wiktextract makes of it."""

    @_SPAWNS
    @given(st.lists(_ENTRY_LINES | _REPORT_LINES, max_size=8), _ENTRY_LINES, st.data())
    def test_filters_process_output(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        lines: list[str],
        entry: str,
        data: st.DataObject,
    ) -> None:
        """Wiktextract reports itself down the same stream as the entries."""
        position = data.draw(st.integers(min_value=0, max_value=len(lines)))
        written = [*lines[:position], entry, *lines[position:]]

        _ = stub_wiktextract(written)

        output_path = workspace() / "wiktextract.jsonl.zst"
        skipped_lines = wiktextract.parse(dump_path, output_path, 1, _preserve_entry)

        entries = [line for line in written if line.startswith("{")]

        assert read(output_path) == entries
        assert skipped_lines == len(written) - len(entries)

    @_SPAWNS
    @given(st.integers(min_value=1, max_value=16))
    def test_forwards_extraction_options(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        processes: int,
    ) -> None:
        """
        The command is compared whole, since it settles what the data is.

        The parser receives the edition language and flags enabling every
        collected resource.
        """
        commands = stub_wiktextract(['{"word": "bank"}'])

        output_path = workspace() / "wiktextract.jsonl.zst"
        _ = wiktextract.parse(dump_path, output_path, processes, _preserve_entry)

        assert commands == [
            [
                "wiktwords",
                "--out",
                "-",
                "--edition",
                LANGUAGE,
                "--language-code",
                LANGUAGE,
                "--examples",
                "--translations",
                "--linkages",
                "--etymologies",
                "--quiet",
                "--num-processes",
                str(processes),
                str(dump_path),
            ],
        ]

    @_SPAWNS
    @given(st.lists(st.sampled_from(["cache", "en", "20260801"]), min_size=1))
    def test_forwards_database_path(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        directories: list[str],
    ) -> None:
        """
        The tail of the command is what is read, the head being settled above.

        A database is asked for by option, so it goes before the dump, which wiktextract
        takes as the one argument that stands on its own.
        """
        commands = stub_wiktextract(['{"word": "bank"}'])

        directory = workspace()
        database_path = directory.joinpath(*directories) / "pages.db"

        _ = wiktextract.parse(
            dump_path,
            directory / "wiktextract.jsonl.zst",
            1,
            _preserve_entry,
            database_path,
        )

        assert commands[0][-3:] == ["--db-path", str(database_path), str(dump_path)]

    @_SPAWNS
    @given(st.lists(st.sampled_from(["cache", "en", "20260801"]), min_size=1))
    def test_creates_parent_directory(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        directories: list[str],
    ) -> None:
        """A parse may be the first thing written where it is going."""
        _ = stub_wiktextract(['{"word": "bank"}'])

        output_path = workspace().joinpath(*directories) / "wiktextract.jsonl.zst"
        _ = wiktextract.parse(dump_path, output_path, 1, _preserve_entry)

        assert output_path.exists()

    @_SPAWNS
    @given(st.lists(_ENTRY_LINES, min_size=1, max_size=4))
    def test_removes_partial_output(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        lines: list[str],
    ) -> None:
        """The .part file is removed once its contents are in place."""
        _ = stub_wiktextract(lines)

        directory = workspace()
        output_path = directory / "wiktextract.jsonl.zst"
        _ = wiktextract.parse(dump_path, output_path, 1, _preserve_entry)

        assert list(directory.iterdir()) == [output_path]

    @_SPAWNS
    @given(st.lists(_REPORT_LINES, max_size=4))
    def test_rejects_empty_extraction(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        lines: list[str],
    ) -> None:
        """An archive of no entries is a file of no bytes, which nothing reads."""
        _ = stub_wiktextract(lines)

        directory = workspace()

        with pytest.raises(RuntimeError, match="no entry"):
            _ = wiktextract.parse(
                dump_path,
                directory / "wiktextract.jsonl.zst",
                1,
                _preserve_entry,
            )

        assert list(directory.iterdir()) == []

    def test_rejects_missing_stdout(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing output stream raises an explicit error."""

        def popen(
            _command: list[str],
            **_keywords: object,
        ) -> _SilentProcess:
            return _SilentProcess()

        monkeypatch.setattr(subprocess, "Popen", popen)

        directory = workspace()

        with pytest.raises(RuntimeError, match="no output to read"):
            _ = wiktextract.parse(
                dump_path,
                directory / "wiktextract.jsonl.zst",
                1,
                _preserve_entry,
            )

        assert list(directory.iterdir()) == []

    @_SPAWNS
    @given(st.lists(_ENTRY_LINES, max_size=4), st.integers(min_value=1, max_value=255))
    def test_discards_failed_extraction(
        self,
        workspace: Callable[[], Path],
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
        lines: list[str],
        return_code: int,
    ) -> None:
        """A run cut short leaves nothing that passes for finished."""
        _ = stub_wiktextract(lines, return_code=return_code)

        directory = workspace()

        with pytest.raises(subprocess.CalledProcessError):
            _ = wiktextract.parse(
                dump_path,
                directory / "wiktextract.jsonl.zst",
                1,
                _preserve_entry,
            )

        assert list(directory.iterdir()) == []
