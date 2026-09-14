"""Tests for src/wsc/upstream/download.py."""

from collections.abc import Callable
from pathlib import Path

import pytest
import requests
import responses
from hypothesis import given
from hypothesis import strategies as st

from wsc.upstream import download

URL = "https://dumps.example.invalid/dump.xml.bz2"
TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"

_BODIES = st.binary(max_size=64)

_CHUNK_SIZES = st.integers(min_value=1, max_value=32)

_REFUSALS = st.sampled_from([400, 403, 404, 429, 500, 503])


def fetch(
    output_path: Path,
    chunk_size: int = 8,
) -> None:
    """
    Download to a path, with the settings every test here shares.

    Args:
        output_path: Where the finished file is placed.
        chunk_size: How much of the response to read at a time.
    """
    download(URL, output_path, USER_AGENT, TIMEOUT, chunk_size)


class TestDownload:
    """Retrieval through a .part file, so a failed attempt can be resumed."""

    @given(_BODIES, _CHUNK_SIZES)
    def test_writes_download(
        self,
        workspace: Callable[[], Path],
        body: bytes,
        chunk_size: int,
    ) -> None:
        """The bytes the server sent are the bytes that land on disk."""
        output_path = workspace() / "dump.xml.bz2"

        with responses.RequestsMock() as server:
            _ = server.get(URL, body=body)

            fetch(output_path, chunk_size)

        assert output_path.read_bytes() == body

    @given(st.lists(st.sampled_from(["cache", "en", "20260801"]), min_size=1))
    def test_creates_parent_directory(
        self,
        workspace: Callable[[], Path],
        directories: list[str],
    ) -> None:
        """The first fetch of an edition writes where no directory exists yet."""
        output_path = workspace().joinpath(*directories) / "dump.xml.bz2"

        with responses.RequestsMock() as server:
            _ = server.get(URL, body=b"a dump")

            fetch(output_path)

        assert output_path.exists()

    @given(_BODIES)
    def test_removes_partial_output(
        self,
        workspace: Callable[[], Path],
        body: bytes,
    ) -> None:
        """The .part file is removed once its contents are in place."""
        directory = workspace()

        with responses.RequestsMock() as server:
            _ = server.get(URL, body=body)

            fetch(directory / "dump.xml.bz2")

        assert list(directory.iterdir()) == [directory / "dump.xml.bz2"]

    def test_sends_package_identity(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        with responses.RequestsMock() as server:
            _ = server.get(URL, body=b"a dump")

            fetch(workspace() / "dump.xml.bz2")

            assert server.calls[0].request.headers["User-Agent"] == USER_AGENT

    def test_omits_initial_range(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """A first attempt has no bytes behind it, so it asks for the whole file."""
        with responses.RequestsMock() as server:
            _ = server.get(URL, body=b"a dump")

            fetch(workspace() / "dump.xml.bz2")

            assert "Range" not in server.calls[0].request.headers

    @given(st.binary(min_size=1, max_size=32), _BODIES, _CHUNK_SIZES)
    def test_resumes_partial_download(
        self,
        workspace: Callable[[], Path],
        downloaded: bytes,
        rest: bytes,
        chunk_size: int,
    ) -> None:
        """A transfer resumes from the .part file a failed attempt left behind."""
        output_path = workspace() / "dump.xml.bz2"
        _ = output_path.with_name("dump.xml.bz2.part").write_bytes(downloaded)

        with responses.RequestsMock() as server:
            _ = server.get(URL, body=rest, status=206)

            fetch(output_path, chunk_size)

            assert server.calls[0].request.headers["Range"] == (
                f"bytes={len(downloaded)}-"
            )

        assert output_path.read_bytes() == downloaded + rest

    @given(st.binary(min_size=1, max_size=32), _BODIES)
    def test_restarts_ignored_range(
        self,
        workspace: Callable[[], Path],
        downloaded: bytes,
        body: bytes,
    ) -> None:
        """A 200 answers with the whole file, so what came before is dropped."""
        output_path = workspace() / "dump.xml.bz2"
        _ = output_path.with_name("dump.xml.bz2.part").write_bytes(downloaded)

        with responses.RequestsMock() as server:
            _ = server.get(URL, body=body, status=200)

            fetch(output_path)

        assert output_path.read_bytes() == body

    @given(st.binary(min_size=1, max_size=32), _REFUSALS)
    def test_preserves_partial_download(
        self,
        workspace: Callable[[], Path],
        downloaded: bytes,
        status: int,
    ) -> None:
        """The .part file left behind is the whole of what resuming builds on."""
        output_path = workspace() / "dump.xml.bz2"
        partial_path = output_path.with_name("dump.xml.bz2.part")
        _ = partial_path.write_bytes(downloaded)

        with responses.RequestsMock() as server:
            _ = server.get(URL, status=status)

            with pytest.raises(requests.HTTPError):
                fetch(output_path)

        assert partial_path.read_bytes() == downloaded

    @given(_REFUSALS)
    def test_propagates_http_errors(
        self,
        workspace: Callable[[], Path],
        status: int,
    ) -> None:
        """A dump that is not there leaves no file behind."""
        directory = workspace()

        with responses.RequestsMock() as server:
            _ = server.get(URL, status=status)

            with pytest.raises(requests.HTTPError):
                fetch(directory / "dump.xml.bz2")

        assert list(directory.iterdir()) == []
