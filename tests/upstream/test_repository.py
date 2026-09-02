"""
Tests for src/wsc/upstream/repository.py.
"""

from collections.abc import Generator, Mapping
from contextlib import contextmanager

import pytest
import requests
import responses
from documents import dump_index, dump_status
from hypothesis import given
from hypothesis import strategies as st
from strategies import dump_dates, languages

from wsc.constants import DUMP_INDEX_URL, DUMP_STATUS_URL
from wsc.upstream import repository

TIMEOUT = (1, 1)
USER_AGENT = "wsc/0.1.0 (https://example.invalid)"

# What one dump answers when asked how far along it is: the state of the job
# the collector waits on, or nothing readable at all.
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
    language: str,
    reports: Mapping[str, str],
    times: int = 1,
) -> Generator[responses.RequestsMock]:
    """
    Answer in Wikimedia's place, for the whole of what a resolution may ask.

    A resolution stops at the first dump it finds finished, so what it never
    asks about is registered all the same and left unasked.

    Args:
        language: Wiktionary's code for the edition.
        reports: What each dump reports, by the day it began: the state of
            its job, or missing for one answering nothing and unreadable for
            one answering something other than a report.
        times: How often the index lists each dump.

    Yields:
        The server, for the calls it took to be read back off.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(
            DUMP_INDEX_URL.format(language=language),
            body=dump_index(*(date for date in reports for _ in range(times))),
        )

        for date, report in reports.items():
            url = DUMP_STATUS_URL.format(language=language, date=date)

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
    """
    Where the archive of pages for one dump sits.
    """

    def test_names_the_archive_after_the_edition_and_the_date(
        self,
    ) -> None:
        """The address is built rather than discovered, so it is built here in full."""
        assert repository.url("en", "20260801") == (
            "https://dumps.wikimedia.org/enwiktionary/20260801/"
            "enwiktionary-20260801-pages-articles.xml.bz2"
        )


class TestLatestDate:
    """
    The most recent dump that has finished being built.
    """

    @given(_LISTINGS, st.data())
    def test_takes_the_newest_dump_that_is_done(
        self,
        reports: dict[str, str],
        data: st.DataObject,
    ) -> None:
        """The newest directory is not the answer: the newest finished one is."""
        reports[data.draw(st.sampled_from(sorted(reports)))] = "done"

        with _wikimedia("en", reports):
            assert repository.latest_date("en", USER_AGENT, TIMEOUT) == _finished(
                reports
            )

    @given(_LISTINGS, st.data())
    def test_asks_about_a_date_once_and_stops_where_it_settles(
        self,
        reports: dict[str, str],
        data: st.DataObject,
    ) -> None:
        """The index names a dump on more than one line, and it is one dump."""
        reports[data.draw(st.sampled_from(sorted(reports)))] = "done"
        times = data.draw(st.integers(min_value=1, max_value=3))

        with _wikimedia("en", reports, times) as server:
            answer = repository.latest_date("en", USER_AGENT, TIMEOUT)

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

        with _wikimedia("en", reports) as server:
            _ = repository.latest_date("en", USER_AGENT, TIMEOUT)

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
            _wikimedia("en", reports),
            pytest.raises(RuntimeError, match="No finished enwiktionary dump"),
        ):
            _ = repository.latest_date("en", USER_AGENT, TIMEOUT)

    @given(languages)
    def test_raises_when_the_index_cannot_be_read(
        self,
        language: str,
    ) -> None:
        """A server that cannot answer is not a server saying there are no dumps."""
        with responses.RequestsMock() as server:
            _ = server.get(DUMP_INDEX_URL.format(language=language), status=503)

            with pytest.raises(requests.HTTPError):
                _ = repository.latest_date(language, USER_AGENT, TIMEOUT)
