"""
Tests for src/wsc/upstream/repositories/wiktionary.py.
"""

import json

import pytest
import requests
import responses

from wsc.constants import DUMP_INDEX_URL, DUMP_STATUS_URL
from wsc.upstream.repositories import wiktionary

TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"


def index(
    *dates: str,
) -> str:
    """
    Write the listing Wikimedia serves for one edition.

    Args:
        dates: The days the dumps it holds began.

    Returns:
        Enough of the page for the date pattern to read it.
    """
    links = "".join(f'<a href="{date}/">{date}/</a>\n' for date in dates)

    return f"<html><body><pre>\n{links}</pre></body></html>"


def status(
    state: str,
) -> str:
    """
    Write what a dump reports about the job the collector waits on.

    Args:
        state: How far along that job is.

    Returns:
        The status file, as JSON.
    """
    return json.dumps({"jobs": {"articlesdumprecombine": {"status": state}}})


class TestUrl:
    """
    Where the archive of pages for one dump sits.
    """

    def test_names_the_archive_after_the_edition_and_the_date(
        self,
    ) -> None:
        """The address is built rather than discovered, so it is built here in full."""
        assert wiktionary.url("en", "20260801") == (
            "https://dumps.wikimedia.org/enwiktionary/20260801/"
            "enwiktionary-20260801-pages-articles.xml.bz2"
        )


class TestLatestDate:
    """
    The most recent dump that has finished being built.
    """

    @responses.activate
    def test_takes_the_newest_dump_that_is_done(
        self,
    ) -> None:
        """The newest directory is not the answer: the newest finished one is."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body=index("20260601", "20260701", "20260801"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=status("done"),
        )

        assert wiktionary.latest_date("en", USER_AGENT, TIMEOUT) == "20260801"

    @responses.activate
    def test_passes_over_a_dump_still_under_way(
        self,
    ) -> None:
        """A dump still being built has a directory too, and is not one to take."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body=index("20260701", "20260801"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=status("in-progress"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260701"),
            body=status("done"),
        )

        assert wiktionary.latest_date("en", USER_AGENT, TIMEOUT) == "20260701"

    @responses.activate
    def test_passes_over_a_dump_reporting_nothing(
        self,
    ) -> None:
        """A dump whose status cannot be read is not known to be finished."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body=index("20260701", "20260801"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            status=404,
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260701"),
            body=status("done"),
        )

        assert wiktionary.latest_date("en", USER_AGENT, TIMEOUT) == "20260701"

    @responses.activate
    def test_passes_over_a_dump_reporting_something_other_than_json(
        self,
    ) -> None:
        """An error page answered with a 200 is still not a report."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body=index("20260701", "20260801"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body="<html><body>Service temporarily unavailable</body></html>",
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260701"),
            body=status("done"),
        )

        assert wiktionary.latest_date("en", USER_AGENT, TIMEOUT) == "20260701"

    @responses.activate
    def test_asks_about_a_date_once_however_often_it_is_listed(
        self,
    ) -> None:
        """The index names a dump on more than one line, and it is one dump."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body=index("20260701", "20260801", "20260801"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=status("in-progress"),
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260701"),
            body=status("done"),
        )

        assert wiktionary.latest_date("en", USER_AGENT, TIMEOUT) == "20260701"
        assert len(responses.calls) == 3

    @responses.activate
    def test_names_itself_to_the_server(
        self,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        _ = responses.get(DUMP_INDEX_URL.format(language="en"), body=index("20260801"))
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=status("done"),
        )

        _ = wiktionary.latest_date("en", USER_AGENT, TIMEOUT)

        assert responses.calls
        assert all(
            call.request.headers["User-Agent"] == USER_AGENT for call in responses.calls
        )

    @responses.activate
    def test_refuses_when_no_dump_has_finished(
        self,
    ) -> None:
        """No dump finished means nothing to fetch, and that is reported."""
        _ = responses.get(DUMP_INDEX_URL.format(language="en"), body=index("20260801"))
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=status("waiting"),
        )

        with pytest.raises(RuntimeError, match="No finished enwiktionary dump"):
            _ = wiktionary.latest_date("en", USER_AGENT, TIMEOUT)

    @responses.activate
    def test_refuses_when_the_index_lists_no_dump(
        self,
    ) -> None:
        """An edition Wikimedia does not publish reads as an empty listing."""
        _ = responses.get(DUMP_INDEX_URL.format(language="xx"), body=index())

        with pytest.raises(RuntimeError):
            _ = wiktionary.latest_date("xx", USER_AGENT, TIMEOUT)

    @responses.activate
    def test_raises_when_the_index_cannot_be_read(
        self,
    ) -> None:
        """A server that cannot answer is not a server saying there are no dumps."""
        _ = responses.get(DUMP_INDEX_URL.format(language="en"), status=503)

        with pytest.raises(requests.HTTPError):
            _ = wiktionary.latest_date("en", USER_AGENT, TIMEOUT)
