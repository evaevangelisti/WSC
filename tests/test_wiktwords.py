"""
Tests for src/wsc/wiktwords.py.

Wiktextract itself is never run: it needs a real dump and the better part of a
day. What is tested is the plumbing around it, and a stand-in subprocess is
enough for that, while keeping real streams and real exit codes.
"""

import subprocess
import sys
from collections.abc import Callable
from compression import zstd
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import pytest

from wsc import wiktwords


class _MutePopen:
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
        """
        Stand in for a process left, which has nothing to release.

        Args:
            exc_type: Class of the exception leaving the block, if any.
            exc_value: The exception itself, if any.
            traceback: Its traceback, if any.
        """


@pytest.fixture
def stub_wiktextract(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[list[str]]]:
    """
    Answer in wiktextract's place, down the stream it would have written to.

    Args:
        monkeypatch: Puts the stand-in in place, and takes it away after.

    Returns:
        A builder taking the lines to write and the code to exit with, and
        handing back the commands it was asked to run.
    """
    real_popen = subprocess.Popen
    commands: list[list[str]] = []

    def build(
        lines: list[str],
        return_code: int = 0,
    ) -> list[list[str]]:
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
    """
    Turning a dump into the compressed JSONL wiktextract makes of it.
    """

    def test_writes_the_entries_compressed(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """A dump's worth of JSON is too much to keep as it comes."""
        _ = stub_wiktextract(['{"word": "bank"}', '{"word": "run"}'])

        output_path = tmp_path / "wiktextract.jsonl.zst"
        _ = wiktwords.parse(dump_path, output_path, "en", 1)

        assert read(output_path) == ['{"word": "bank"}', '{"word": "run"}']

    def test_sets_aside_what_is_not_an_entry(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """Wiktextract reports itself down the same stream as the entries."""
        _ = stub_wiktextract(
            ["Parsing pages...", '{"word": "bank"}', "Done in 3h."],
        )

        output_path = tmp_path / "wiktextract.jsonl.zst"
        skipped_lines = wiktwords.parse(dump_path, output_path, "en", 1)

        assert skipped_lines == 2
        assert read(output_path) == ['{"word": "bank"}']

    def test_asks_wiktextract_for_what_the_collector_reads(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """
        The command is compared whole, since it settles what the data is.

        The edition to read and the language to keep are the same one, and
        --examples is what puts the sentences in the output at all.
        """
        commands = stub_wiktextract([])

        _ = wiktwords.parse(dump_path, tmp_path / "out.jsonl.zst", "it", 4)

        assert commands == [
            [
                "wiktwords",
                "--out",
                "-",
                "--edition",
                "it",
                "--language-code",
                "it",
                "--examples",
                "--num-processes",
                "4",
                str(dump_path),
            ]
        ]

    def test_creates_the_parent_directory(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """A parse may be the first thing written where it is going."""
        _ = stub_wiktextract(['{"word": "bank"}'])

        output_path = tmp_path / "deep" / "wiktextract.jsonl.zst"
        _ = wiktwords.parse(dump_path, output_path, "en", 1)

        assert output_path.exists()

    def test_leaves_no_partial_file_behind(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """The .part file is removed once its contents are in place."""
        _ = stub_wiktextract(['{"word": "bank"}'])

        output_path = tmp_path / "out" / "wiktextract.jsonl.zst"
        _ = wiktwords.parse(dump_path, output_path, "en", 1)

        assert list(output_path.parent.iterdir()) == [output_path]

    def test_raises_when_wiktextract_offers_no_output(
        self,
        tmp_path: Path,
        dump_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A process with no stream to read is reported, not read from anyway."""

        def popen(
            _command: list[str],
            **_keywords: object,
        ) -> _MutePopen:
            return _MutePopen()

        monkeypatch.setattr(subprocess, "Popen", popen)

        output_path = tmp_path / "out" / "wiktextract.jsonl.zst"

        with pytest.raises(RuntimeError, match="no output to read"):
            _ = wiktwords.parse(dump_path, output_path, "en", 1)

        assert list(output_path.parent.iterdir()) == []

    def test_raises_when_wiktextract_answers_with_an_error(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """An error from wiktextract is raised rather than passed over."""
        _ = stub_wiktextract(['{"word": "bank"}'], return_code=1)

        output_path = tmp_path / "wiktextract.jsonl.zst"

        with pytest.raises(subprocess.CalledProcessError):
            _ = wiktwords.parse(dump_path, output_path, "en", 1)

    def test_writes_nothing_when_wiktextract_fails(
        self,
        tmp_path: Path,
        dump_path: Path,
        stub_wiktextract: Callable[..., list[list[str]]],
    ) -> None:
        """A run cut short leaves nothing that passes for finished."""
        _ = stub_wiktextract(['{"word": "bank"}'], return_code=1)

        output_path = tmp_path / "out" / "wiktextract.jsonl.zst"

        with pytest.raises(subprocess.CalledProcessError):
            _ = wiktwords.parse(dump_path, output_path, "en", 1)

        assert list(output_path.parent.iterdir()) == []
