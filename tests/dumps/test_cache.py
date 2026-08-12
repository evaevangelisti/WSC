"""
Tests for src/wsc/dumps/cache.py.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from wsc.dumps import cache


class TestDumpDir:
    """
    Naming the directory one dump sits in.
    """

    def test_names_the_directory_after_the_edition_and_the_date(
        self,
        cache_dir: Path,
    ) -> None:
        """The name carries both, since that is what tells two dumps apart."""
        assert (
            cache.dump_dir(cache_dir, "en", "20260801")
            == cache_dir / "enwiktionary-20260801"
        )

    def test_creates_nothing(
        self,
        cache_dir: Path,
    ) -> None:
        """Naming a dump is not fetching one, so the disk is left alone."""
        _ = cache.dump_dir(cache_dir, "en", "20260801")

        assert not cache_dir.exists()

    def test_falls_back_to_the_platform_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """None is what the command line passes when no cache was named."""

        def user_cache_dir(
            appname: str,
        ) -> str:
            return str(tmp_path / appname)

        monkeypatch.setattr(cache, "user_cache_dir", user_cache_dir)

        assert cache.dump_dir(None, "en", "20260801").parent == tmp_path / "wsc"


class TestFetchedDate:
    """
    Settling which fetched dump to work on.
    """

    def test_accepts_a_date_that_was_fetched(
        self,
        cache_dir: Path,
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A dump is known by the directory it sits in, and that one is there."""
        _ = fetch_dump("en", "20260801")

        assert cache.fetched_date(cache_dir, "en", "20260801") == "20260801"

    def test_refuses_a_date_that_was_not_fetched(
        self,
        cache_dir: Path,
        fetch_dump: Callable[..., Path],
    ) -> None:
        """The date asked for is named in the refusal, since it is the one to fetch."""
        _ = fetch_dump("en", "20260801")

        with pytest.raises(FileNotFoundError, match="20260701"):
            _ = cache.fetched_date(cache_dir, "en", "20260701")

    def test_latest_takes_the_newest_dump_fetched(
        self,
        cache_dir: Path,
        fetch_dump: Callable[..., Path],
    ) -> None:
        """Latest is answered from the cache, so a machine offline runs the same."""
        for date in ("20260601", "20260801", "20260701"):
            _ = fetch_dump("en", date)

        assert cache.fetched_date(cache_dir, "en", cache.LATEST) == "20260801"

    def test_latest_reads_the_edition_asked_for_alone(
        self,
        cache_dir: Path,
        fetch_dump: Callable[..., Path],
    ) -> None:
        """One cache holds every edition, and each is fetched on its own."""
        _ = fetch_dump("it", "20260901")
        _ = fetch_dump("en", "20260801")

        assert cache.fetched_date(cache_dir, "en", cache.LATEST) == "20260801"

    def test_latest_passes_over_a_file_named_like_a_dump(
        self,
        cache_dir: Path,
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A dump is a directory, so whatever else lands there is not one."""
        _ = fetch_dump("en", "20260801")
        _ = (cache_dir / "enwiktionary-20260901").write_text("not a dump")

        assert cache.fetched_date(cache_dir, "en", cache.LATEST) == "20260801"

    def test_refuses_latest_when_nothing_was_fetched(
        self,
        cache_dir: Path,
    ) -> None:
        """An empty cache is reported, and the report says what to do about it."""
        with pytest.raises(FileNotFoundError, match="fetch one first"):
            _ = cache.fetched_date(cache_dir, "en", cache.LATEST)

    def test_refuses_latest_when_the_cache_does_not_exist(
        self,
        cache_dir: Path,
    ) -> None:
        """A cache nobody has written to yet reads as an empty one."""
        with pytest.raises(FileNotFoundError):
            _ = cache.fetched_date(cache_dir / "missing", "en", cache.LATEST)
