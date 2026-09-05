"""
Tests for src/wsc/upstream/cache.py.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from platformdirs import user_cache_dir
from strategies import dump_dates

from wsc.upstream import cache

_FETCHED = st.lists(dump_dates, min_size=1, max_size=4, unique=True)


class TestDumpDir:
    """
    Naming the directory one dump sits in.
    """

    @given(dump_dates)
    def test_names_the_directory_after_the_date(
        self,
        workspace: Callable[[], Path],
        date: str,
    ) -> None:
        """One directory per source, then per dump."""
        cache_dir = workspace()

        assert cache.dump_dir(cache_dir, date) == cache_dir / "wiktionary" / date

    @given(dump_dates)
    def test_falls_back_to_the_platform_cache(
        self,
        date: str,
    ) -> None:
        """None is what the command line passes when no cache was named."""
        assert cache.dump_dir(None, date) == (
            Path(user_cache_dir("wsc")) / "wiktionary" / date
        )

    @given(dump_dates)
    def test_creates_nothing(
        self,
        workspace: Callable[[], Path],
        date: str,
    ) -> None:
        """Naming a dump is not fetching one, so the disk is left alone."""
        directory = workspace()

        _ = cache.dump_dir(directory / "cache", date)

        assert list(directory.iterdir()) == []


class TestFetchedDate:
    """
    Settling which fetched dump to work on.
    """

    @given(_FETCHED)
    def test_accepts_every_date_that_was_fetched(
        self,
        workspace: Callable[[], Path],
        fetch_dump: Callable[..., Path],
        dates: list[str],
    ) -> None:
        """A dump is known by the directory it sits in, and those are there."""
        cache_dir = workspace()

        for date in dates:
            _ = fetch_dump(cache_dir, date)

        assert [cache.fetched_date(cache_dir, date) for date in dates] == dates

    @given(_FETCHED)
    def test_latest_takes_the_newest_dump_fetched(
        self,
        workspace: Callable[[], Path],
        fetch_dump: Callable[..., Path],
        dates: list[str],
    ) -> None:
        """Latest is answered from the cache, so a machine offline runs the same."""
        cache_dir = workspace()

        for date in dates:
            _ = fetch_dump(cache_dir, date)

        assert cache.fetched_date(cache_dir, cache.LATEST) == max(dates)

    @given(_FETCHED, st.data())
    def test_latest_passes_over_a_file_named_like_a_dump(
        self,
        workspace: Callable[[], Path],
        fetch_dump: Callable[..., Path],
        dates: list[str],
        data: st.DataObject,
    ) -> None:
        """A dump is a directory, so whatever else lands there is not one."""
        cache_dir = workspace()

        for date in dates:
            _ = fetch_dump(cache_dir, date)

        newer = data.draw(dump_dates.filter(lambda date: date > max(dates)))
        _ = (cache_dir / "wiktionary" / newer).write_text("not a dump")

        assert cache.fetched_date(cache_dir, cache.LATEST) == max(dates)

    @given(_FETCHED, st.data())
    def test_refuses_a_date_that_was_not_fetched(
        self,
        workspace: Callable[[], Path],
        fetch_dump: Callable[..., Path],
        dates: list[str],
        data: st.DataObject,
    ) -> None:
        """The date asked for is named in the refusal, since it is the one to fetch."""
        cache_dir = workspace()

        for date in dates:
            _ = fetch_dump(cache_dir, date)

        missing = data.draw(dump_dates.filter(lambda date: date not in dates))

        with pytest.raises(FileNotFoundError, match=missing):
            _ = cache.fetched_date(cache_dir, missing)

    def test_refuses_latest_when_nothing_was_fetched(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """An empty cache is reported, and the report says what to do about it."""
        with pytest.raises(FileNotFoundError, match="fetch one first"):
            _ = cache.fetched_date(workspace(), cache.LATEST)

    def test_refuses_latest_when_the_cache_does_not_exist(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """A cache nobody has written to yet reads as an empty one."""
        with pytest.raises(FileNotFoundError):
            _ = cache.fetched_date(workspace() / "missing", cache.LATEST)
