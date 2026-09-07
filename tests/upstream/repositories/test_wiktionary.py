"""Tests for src/wsc/upstream/repositories/wiktionary.py."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager

import pytest
import requests
import responses
from documents import dump_index, dump_status
from hypothesis import given
from hypothesis import strategies as st
from strategies import dump_dates

from wsc.constants import DUMP_INDEX_URL, DUMP_STATUS_URL
from wsc.upstream.repositories import wiktionary

TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"

_REPORTS = st.sampled_from(
    ["done", "in-progress", "waiting", "skipped", "missing", "unreadable"]
)

_LISTINGS = st.dictionaries(dump_dates, _REPORTS, min_size=1, max_size=5)

_UNFINISHED = st.dictionaries(
    dump_dates,
    _REPORTS.filter(lambda report: report != "done"),
    max_size=4,
)


@contextmanager
def _wikimedia(
    reports: Mapping[str, str],
    times: int = 1,
) -> Generator[responses.RequestsMock]:
    """
    Answer in Wikimedia's place, for the whole of what a resolution may ask.

    A resolution stops at the first dump it finds finished, so what it never asks about
    is registered all the same and left unasked.

    Args:
        reports: Dump dates mapped to job states or invalid response markers.
        times: How often the index lists each dump.

    Yields:
        The server, for the calls it took to be read back off.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(
            DUMP_INDEX_URL,
            body=dump_index(*(date for date in reports for _ in range(times))),
        )

        for date, report in reports.items():
            url = DUMP_STATUS_URL.format(date=date)

            match report:
                case "missing":
                    _ = server.get(url, status=404)

                case "unreadable":
                    _ = server.get(url, body="<html><body>Come back</body></html>")

                case _:
                    _ = server.get(url, body=dump_status(report))

        yield server


def _finished(
    reports: Mapping[str, str],
) -> str:
    """
    Name the dump a resolution is expected to settle on.

    Args:
        reports: What each dump reports, by the day it began.

    Returns:
        The newest day whose dump reports itself done.
    """
    return max(date for date, report in reports.items() if report == "done")


class TestUrl:
    """Where the archive of pages for one dump sits."""

    def test_names_the_archive_after_the_date(
        self,
    ) -> None:
        """The generated address includes the requested edition and filename."""
        assert wiktionary.url("20260801") == (
            "https://dumps.wikimedia.org/enwiktionary/20260801/"
            "enwiktionary-20260801-pages-articles.xml.bz2"
        )


class TestLatestDate:
    """The most recent dump that has finished being built."""

    @given(_LISTINGS, st.data())
    def test_takes_the_newest_dump_that_is_done(
        self,
        reports: dict[str, str],
        data: st.DataObject,
    ) -> None:
        """The newest directory is not the answer: the newest finished one is."""
        reports[data.draw(st.sampled_from(sorted(reports)))] = "done"

        with _wikimedia(reports):
            assert wiktionary.latest_date(USER_AGENT, TIMEOUT) == _finished(reports)

    @given(_LISTINGS, st.data())
    def test_asks_about_a_date_once_and_stops_where_it_settles(
        self,
        reports: dict[str, str],
        data: st.DataObject,
    ) -> None:
        """The index names a dump on more than one line, and it is one dump."""
        reports[data.draw(st.sampled_from(sorted(reports)))] = "done"
        times = data.draw(st.integers(min_value=1, max_value=3))

        with _wikimedia(reports, times) as server:
            answer = wiktionary.latest_date(USER_AGENT, TIMEOUT)

            asked = [
                call
                for call in server.calls
                if str(call.request.url).endswith("dumpstatus.json")
            ]

        assert len(asked) == len([date for date in reports if date > answer]) + 1

    @given(_LISTINGS, st.data())
    def test_names_itself_to_the_server(
        self,
        reports: dict[str, str],
        data: st.DataObject,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        reports[data.draw(st.sampled_from(sorted(reports)))] = "done"

        with _wikimedia(reports) as server:
            _ = wiktionary.latest_date(USER_AGENT, TIMEOUT)

            assert server.calls
            assert all(
                call.request.headers["User-Agent"] == USER_AGENT
                for call in server.calls
            )

    @given(_UNFINISHED)
    def test_refuses_when_no_dump_has_finished(
        self,
        reports: dict[str, str],
    ) -> None:
        """No dump finished, or none listed at all, means nothing to fetch."""
        with (
            _wikimedia(reports),
            pytest.raises(RuntimeError, match="No finished dump"),
        ):
            _ = wiktionary.latest_date(USER_AGENT, TIMEOUT)

    def test_raises_when_the_index_cannot_be_read(
        self,
    ) -> None:
        """A server that cannot answer is not a server saying there are no dumps."""
        with responses.RequestsMock() as server:
            _ = server.get(DUMP_INDEX_URL, status=503)

            with pytest.raises(requests.HTTPError):
                _ = wiktionary.latest_date(USER_AGENT, TIMEOUT)
