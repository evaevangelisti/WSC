"""
Tests for src/wsc/upstream/download.py.
"""

from pathlib import Path

import pytest
import requests
import responses

from wsc.upstream.download import download

URL = "https://dumps.example.invalid/dump.xml.bz2"
TIMEOUT = (1, 1)
CHUNK_SIZE = 8
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"


def fetch(
    output_path: Path,
) -> None:
    """
    Download to a path, with the settings every test here shares.

    Args:
        output_path: Where the finished file is placed.
    """
    download(URL, output_path, USER_AGENT, TIMEOUT, CHUNK_SIZE)


class TestDownload:
    """
    Retrieval through a .part file, so a failed attempt can be resumed.
    """

    @responses.activate
    def test_writes_the_file(
        self,
        tmp_path: Path,
    ) -> None:
        """The bytes the server sent are the bytes that land on disk."""
        _ = responses.get(URL, body=b"a dump, in full")

        output_path = tmp_path / "dump.xml.bz2"
        fetch(output_path)

        assert output_path.read_bytes() == b"a dump, in full"

    @responses.activate
    def test_creates_the_parent_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """The first fetch of an edition writes where no directory exists yet."""
        _ = responses.get(URL, body=b"a dump")

        output_path = tmp_path / "cache" / "enwiktionary-20260801" / "dump.xml.bz2"
        fetch(output_path)

        assert output_path.exists()

    @responses.activate
    def test_leaves_no_partial_file_behind(
        self,
        tmp_path: Path,
    ) -> None:
        """The .part file is removed once its contents are in place."""
        _ = responses.get(URL, body=b"a dump")

        output_path = tmp_path / "dump.xml.bz2"
        fetch(output_path)

        assert list(tmp_path.iterdir()) == [output_path]

    @responses.activate
    def test_names_itself_to_the_server(
        self,
        tmp_path: Path,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        _ = responses.get(URL, body=b"a dump")

        fetch(tmp_path / "dump.xml.bz2")

        assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT

    @responses.activate
    def test_asks_for_nothing_when_there_is_nothing_to_resume(
        self,
        tmp_path: Path,
    ) -> None:
        """A first attempt has no bytes behind it, so it asks for the whole file."""
        _ = responses.get(URL, body=b"a dump")

        fetch(tmp_path / "dump.xml.bz2")

        assert "Range" not in responses.calls[0].request.headers

    @responses.activate
    def test_resumes_from_what_a_failed_attempt_left(
        self,
        tmp_path: Path,
    ) -> None:
        """A transfer resumes from the .part file a failed attempt left behind."""
        output_path = tmp_path / "dump.xml.bz2"
        _ = output_path.with_name("dump.xml.bz2.part").write_bytes(b"a dump, ")

        _ = responses.get(URL, body=b"in full", status=206)

        fetch(output_path)

        assert responses.calls[0].request.headers["Range"] == "bytes=8-"
        assert output_path.read_bytes() == b"a dump, in full"

    @responses.activate
    def test_starts_over_when_the_server_ignores_the_range(
        self,
        tmp_path: Path,
    ) -> None:
        """A 200 answers with the whole file, so what came before is dropped."""
        output_path = tmp_path / "dump.xml.bz2"
        _ = output_path.with_name("dump.xml.bz2.part").write_bytes(b"stale bytes")

        _ = responses.get(URL, body=b"a dump, in full", status=200)

        fetch(output_path)

        assert output_path.read_bytes() == b"a dump, in full"

    @responses.activate
    def test_keeps_what_a_failed_attempt_had_downloaded(
        self,
        tmp_path: Path,
    ) -> None:
        """The .part file left behind is the whole of what resuming builds on."""
        output_path = tmp_path / "dump.xml.bz2"
        partial_path = output_path.with_name("dump.xml.bz2.part")
        _ = partial_path.write_bytes(b"a dump, ")

        _ = responses.get(URL, status=503)

        with pytest.raises(requests.HTTPError):
            fetch(output_path)

        assert partial_path.read_bytes() == b"a dump, "

    @responses.activate
    def test_raises_when_the_server_refuses(
        self,
        tmp_path: Path,
    ) -> None:
        """A dump that is not there leaves no file behind."""
        _ = responses.get(URL, status=404)

        output_path = tmp_path / "dump.xml.bz2"

        with pytest.raises(requests.HTTPError):
            fetch(output_path)

        assert not output_path.exists()
