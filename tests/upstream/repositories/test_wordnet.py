"""
Tests for src/wsc/upstream/repositories/wordnet.py.
"""

import pytest
import requests
import responses
from documents import wordnet_index
from hypothesis import given
from hypothesis import strategies as st
from strategies import wordnet_versions

from wsc.constants import WORDNET_INDEX_URL
from wsc.upstream.repositories import wordnet

TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"

_EDITIONS = st.lists(wordnet_versions, min_size=1, max_size=4, unique=True)


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

    @given(_EDITIONS)
    def test_takes_the_newest_the_index_lists(
        self,
        versions: list[str],
    ) -> None:
        """Editions are named after their year, so the highest is the newest."""
        with responses.RequestsMock() as server:
            _ = server.get(WORDNET_INDEX_URL, body=wordnet_index(*versions))

            assert wordnet.latest_version(USER_AGENT, TIMEOUT) == max(versions)

    @given(_EDITIONS)
    def test_names_the_collector_to_the_server(
        self,
        versions: list[str],
    ) -> None:
        """A download says who answers for it wherever it is sent."""
        with responses.RequestsMock() as server:
            _ = server.get(WORDNET_INDEX_URL, body=wordnet_index(*versions))

            _ = wordnet.latest_version(USER_AGENT, TIMEOUT)

            assert server.calls[0].request.headers["User-Agent"] == USER_AGENT

    def test_raises_when_the_index_lists_no_edition(
        self,
    ) -> None:
        """A page that answered but holds nothing is as good as no page."""
        with responses.RequestsMock() as server:
            _ = server.get(WORDNET_INDEX_URL, body=wordnet_index())

            with pytest.raises(RuntimeError):
                _ = wordnet.latest_version(USER_AGENT, TIMEOUT)

    @given(st.sampled_from([403, 404, 500, 503]))
    def test_raises_when_the_index_cannot_be_read(
        self,
        status: int,
    ) -> None:
        """Nothing downstream can stand in for an index that did not answer."""
        with responses.RequestsMock() as server:
            _ = server.get(WORDNET_INDEX_URL, status=status)

            with pytest.raises(requests.HTTPError):
                _ = wordnet.latest_version(USER_AGENT, TIMEOUT)
