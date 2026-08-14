"""
Tests for src/wsc/upstream/repositories/wordnet.py.
"""

import pytest
import requests
import responses

from wsc.constants import WORDNET_INDEX_URL
from wsc.upstream.repositories import wordnet

TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"


def index(
    *versions: str,
) -> str:
    """
    Write the listing the wordnet serves for its editions.

    Args:
        versions: The editions it holds.

    Returns:
        Enough of the page for the version pattern to read it, every edition
        being offered in the three formats the real page offers.
    """
    links = "".join(
        f'<a href="/downloads/english-wordnet-{version}.{suffix}">{suffix}</a>\n'
        for version in versions
        for suffix in ("ttl.gz", "xml.gz", "zip")
    )

    return f"<html><body>\n{links}</body></html>"


class TestUrl:
    """
    Where one edition of the wordnet sits.
    """

    def test_names_the_file_after_the_edition(
        self,
    ) -> None:
        """The address is built rather than discovered, so it is built here in full."""
        assert wordnet.url("2025") == (
            "https://en-word.net/downloads/english-wordnet-2025.xml.gz"
        )


class TestLatestVersion:
    """
    The most recent edition of the wordnet.
    """

    @responses.activate
    def test_takes_the_newest_the_index_lists(
        self,
    ) -> None:
        """Editions are named after their year, so the highest is the newest."""
        _ = responses.get(WORDNET_INDEX_URL, body=index("2023", "2025", "2024"))

        assert wordnet.latest_version(USER_AGENT, TIMEOUT) == "2025"

    @responses.activate
    def test_reads_one_edition_once_however_often_it_is_offered(
        self,
    ) -> None:
        """The page lists every edition three times, once per format."""
        _ = responses.get(WORDNET_INDEX_URL, body=index("2025"))

        assert wordnet.latest_version(USER_AGENT, TIMEOUT) == "2025"

    @responses.activate
    def test_names_the_collector_to_the_server(
        self,
    ) -> None:
        """A download says who answers for it wherever it is sent."""
        _ = responses.get(WORDNET_INDEX_URL, body=index("2025"))

        _ = wordnet.latest_version(USER_AGENT, TIMEOUT)

        assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT

    @responses.activate
    def test_raises_when_the_index_cannot_be_read(
        self,
    ) -> None:
        """Nothing downstream can stand in for an index that did not answer."""
        _ = responses.get(WORDNET_INDEX_URL, status=503)

        with pytest.raises(requests.HTTPError):
            _ = wordnet.latest_version(USER_AGENT, TIMEOUT)

    @responses.activate
    def test_raises_when_the_index_lists_no_edition(
        self,
    ) -> None:
        """A page that answered but holds nothing is as good as no page."""
        _ = responses.get(WORDNET_INDEX_URL, body="<html><body></body></html>")

        with pytest.raises(RuntimeError):
            _ = wordnet.latest_version(USER_AGENT, TIMEOUT)
