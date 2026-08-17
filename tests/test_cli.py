"""
Tests for src/wsc/cli.py.

A command is tested for what it settles and what it refuses, not for what the
modules under it already answer for: one dump is served here, and which dump
that would be among many is settled where the repository is tested. Only
collect is run end to end, since it is the one command that reaches the disk
without reaching the network.
"""

import json
import string
from collections.abc import Callable, Generator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import cast

import pytest
import responses
from documents import dump_index, dump_status, wordnet_index
from hypothesis import given
from hypothesis import strategies as st
from strategies import (
    RawJson,
    dump_dates,
    languages,
    parts_of_speech,
    raw_entries,
    raw_examples,
    raw_senses,
    references,
    wordnet_versions,
    years,
)
from typer.testing import CliRunner, Result

from wsc.cli import app
from wsc.constants import (
    DUMP_INDEX_URL,
    DUMP_STATUS_URL,
    DUMP_URL,
    USER_AGENT,
    WORDNET_INDEX_URL,
    WORDNET_URL,
)
from wsc.upstream import cache, wiktextract

# A suffix naming no format the collector writes.
_SUFFIXES = st.text(
    alphabet=string.ascii_lowercase,
    min_size=1,
    max_size=6,
).filter(lambda suffix: suffix != "jsonl")


def _said(
    result: Result,
) -> str:
    """
    Read what a command said, with the wrapping taken out.

    Args:
        result: What the run handed back.

    Returns:
        The output as a single line, since where a refusal is broken across
        lines, and what border it is broken around, is the terminal's
        business rather than the command's.
    """
    return " ".join(result.output.replace("│", " ").split())


def _years_of(
    record: RawJson,
) -> list[object]:
    """
    Read the years off the sentences one collected lemma carries.

    The schema is what the JSONL writer is tested on; here it is only walked.

    Args:
        record: One lemma, as a collection wrote it.

    Returns:
        The year of each sentence of its first sense.
    """
    senses = cast(list[RawJson], record["senses"])
    sentences = cast(list[RawJson], senses[0]["sentences"])

    return [sentence["year"] for sentence in sentences]


@contextmanager
def _wikimedia(
    language: str,
    date: str,
) -> Generator[responses.RequestsMock]:
    """
    Answer in Wikimedia's place: one edition, one dump, and it is finished.

    Args:
        language: Wiktionary's code for the edition.
        date: The day the dump it holds began.

    Yields:
        The server, for the calls it took to be read back off. A command that
        was told which dump to fetch leaves the index unasked.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(
            DUMP_INDEX_URL.format(language=language),
            body=dump_index(date),
        )
        _ = server.get(
            DUMP_STATUS_URL.format(language=language, date=date),
            body=dump_status("done"),
        )
        _ = server.get(
            DUMP_URL.format(language=language, date=date),
            body=b"a dump",
        )

        yield server


@contextmanager
def _en_word_net(
    version: str,
) -> Generator[responses.RequestsMock]:
    """
    Answer in the wordnet's place, which is published away from Wikimedia.

    Args:
        version: The edition it holds.

    Yields:
        The server, for the calls it took to be read back off. A command that
        was told which edition to fetch leaves the index unasked.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(WORDNET_INDEX_URL, body=wordnet_index(version))
        _ = server.get(WORDNET_URL.format(version=version), body=b"a wordnet")

        yield server


@pytest.fixture
def cli() -> Callable[..., Result]:
    """
    Run a command against the cache the caller set aside.

    Returns:
        A runner appending the cache, so that no run reads another's dumps.
    """
    runner = CliRunner()

    def invoke(
        *arguments: str,
        cache_dir: Path,
        env: dict[str, str] | None = None,
    ) -> Result:
        return runner.invoke(
            app,
            [*arguments, "--cache-dir", str(cache_dir)],
            env=env,
        )

    return invoke


@pytest.fixture
def stub_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[tuple[Path, Path, str, int, Path | None]]]:
    """
    Answer in wiktextract's place, which its own module is tested on.

    Args:
        monkeypatch: Puts the stand-in in place, and takes it away after.

    Returns:
        A builder taking the lines to report as set aside, and handing back
        the calls one run went on to make.
    """

    def build(
        skipped_lines: int = 0,
    ) -> list[tuple[Path, Path, str, int, Path | None]]:
        calls: list[tuple[Path, Path, str, int, Path | None]] = []

        def parse(
            dump_path: Path,
            output_path: Path,
            language: str,
            processes: int,
            database_path: Path | None,
        ) -> int:
            calls.append((dump_path, output_path, language, processes, database_path))

            return skipped_lines

        monkeypatch.setattr(wiktextract, "parse", parse)

        return calls

    return build


@pytest.fixture
def collected() -> Callable[[Path], list[RawJson]]:
    """
    Read back what a collection wrote.

    Returns:
        A reader handing back one decoded object per line, splitting where
        JSON Lines splits and nowhere else.
    """

    def read(
        output_path: Path,
    ) -> list[RawJson]:
        text = output_path.read_text(encoding="utf-8")

        return [json.loads(line) for line in text.split("\n") if line]

    return read


class TestFetch:
    """
    Downloading a dump, and settling which one that is.
    """

    @given(languages, dump_dates)
    def test_resolves_latest_against_wikimedia(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        language: str,
        date: str,
    ) -> None:
        """Latest is the newest finished dump, and only Wikimedia knows which."""
        cache_dir = workspace() / "cache"

        with _wikimedia(language, date):
            result = cli("fetch", "--language", language, cache_dir=cache_dir)

        assert result.exit_code == 0
        assert f"Resolved latest to {date}" in _said(result)
        assert (
            cache.dump_dir(cache_dir, language, date) / cache.DUMP_NAME
        ).read_bytes() == b"a dump"

    @given(languages, dump_dates)
    def test_fetches_the_dump_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        language: str,
        date: str,
    ) -> None:
        """A dump named on the command line is fetched without asking the index."""
        cache_dir = workspace() / "cache"

        with _wikimedia(language, date) as server:
            result = cli(
                "fetch",
                "--language",
                language,
                "--dump-date",
                date,
                cache_dir=cache_dir,
            )

            assert [call.request.url for call in server.calls] == [
                DUMP_URL.format(language=language, date=date)
            ]

        assert result.exit_code == 0

    @given(languages, dump_dates)
    def test_names_the_collector_and_its_version_to_wikimedia(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        language: str,
        date: str,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        cache_dir = workspace() / "cache"
        expected = USER_AGENT.format(version=version("wsc"))

        with _wikimedia(language, date) as server:
            _ = cli("fetch", "--language", language, cache_dir=cache_dir)

            assert server.calls
            assert all(
                call.request.headers["User-Agent"] == expected for call in server.calls
            )

    @given(languages, dump_dates)
    def test_stops_when_what_it_would_fetch_is_already_there(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        language: str,
        date: str,
    ) -> None:
        """A dump is tens of gigabytes, and is not downloaded twice."""
        cache_dir = workspace() / "cache"
        _ = fetch_dump(cache_dir, language, date)

        with responses.RequestsMock() as server:
            result = cli(
                "fetch",
                "--language",
                language,
                "--dump-date",
                date,
                cache_dir=cache_dir,
            )

            assert not server.calls

        assert result.exit_code == 0
        assert "Already fetched" in _said(result)


class TestParse:
    """
    What a parse settles, wiktextract standing in for itself.
    """

    @given(languages, dump_dates, st.integers(min_value=1, max_value=16))
    def test_points_wiktextract_at_what_the_command_settled(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int, Path | None]]],
        language: str,
        date: str,
        processes: int,
    ) -> None:
        """The dump to read, the file to write and the options are handed over."""
        cache_dir = workspace() / "cache"

        calls = stub_parse()
        _ = fetch_dump(cache_dir, language, date)

        result = cli(
            "parse",
            "--language",
            language,
            "--processes",
            str(processes),
            cache_dir=cache_dir,
        )

        dump_dir = cache.dump_dir(cache_dir, language, date)

        assert result.exit_code == 0
        assert calls == [
            (
                dump_dir / cache.DUMP_NAME,
                dump_dir / cache.WIKTEXTRACT_NAME,
                language,
                processes,
                None,
            )
        ]
        assert "Parsed" in _said(result)

    @given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=8))
    def test_hands_over_the_database_it_was_told_to_keep(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int, Path | None]]],
        name: str,
    ) -> None:
        """A database of one's own is what a run without the network needs."""
        directory = workspace()
        cache_dir = directory / "cache"

        calls = stub_parse()
        _ = fetch_dump(cache_dir)

        database_path = directory / f"{name}.db"
        result = cli(
            "parse",
            "--db-path",
            str(database_path),
            cache_dir=cache_dir,
        )

        assert result.exit_code == 0
        assert calls[0][4] == database_path

    @given(st.lists(dump_dates, min_size=2, max_size=4, unique=True), st.data())
    def test_parses_the_dump_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int, Path | None]]],
        dates: list[str],
        data: st.DataObject,
    ) -> None:
        """A dump named on the command line is the one parsed, latest or not."""
        cache_dir = workspace() / "cache"

        calls = stub_parse()
        for date in dates:
            _ = fetch_dump(cache_dir, "en", date)

        asked = data.draw(st.sampled_from(dates))
        _ = cli("parse", "--dump-date", asked, cache_dir=cache_dir)

        assert calls[0][0] == cache.dump_dir(cache_dir, "en", asked) / cache.DUMP_NAME

    @given(st.integers(min_value=1, max_value=10000))
    def test_reports_the_lines_set_aside(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int, Path | None]]],
        skipped_lines: int,
    ) -> None:
        """Far more than a few hundred means something went wrong, so it is said."""
        cache_dir = workspace() / "cache"

        _ = stub_parse(skipped_lines=skipped_lines)
        _ = fetch_dump(cache_dir)

        result = cli("parse", cache_dir=cache_dir)

        assert f"Set aside {skipped_lines} lines" in _said(result)

    def test_says_nothing_when_no_line_was_set_aside(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int, Path | None]]],
    ) -> None:
        """A parse with nothing to report reports nothing."""
        cache_dir = workspace() / "cache"

        _ = stub_parse()
        _ = fetch_dump(cache_dir)

        result = cli("parse", cache_dir=cache_dir)

        assert "Set aside" not in _said(result)

    @given(st.lists(raw_entries(), max_size=3))
    def test_stops_when_the_dump_was_already_parsed(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        entries: list[RawJson],
    ) -> None:
        """A parse takes the better part of a day, and is not repeated for nothing."""
        cache_dir = workspace() / "cache"
        _ = parse_dump(cache_dir, entries)

        result = cli("parse", cache_dir=cache_dir)

        assert result.exit_code == 0
        assert "Already parsed" in _said(result)

    @given(languages)
    def test_refuses_when_the_dump_was_not_fetched(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        language: str,
    ) -> None:
        """Parsing reads a dump, so there has to be one to read."""
        result = cli(
            "parse",
            "--language",
            language,
            cache_dir=workspace() / "cache",
        )

        assert result.exit_code != 0
        assert "fetch one first" in _said(result)

    @given(languages, dump_dates)
    def test_refuses_when_the_dump_is_gone_from_its_directory(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        language: str,
        date: str,
    ) -> None:
        """A fetch cut short leaves the directory behind without the dump in it."""
        cache_dir = workspace() / "cache"
        cache.dump_dir(cache_dir, language, date).mkdir(parents=True)

        result = cli("parse", "--language", language, cache_dir=cache_dir)

        assert result.exit_code != 0
        assert "fetch it first" in _said(result)


class TestCollect:
    """
    Collecting the senses of a parsed dump into a file.
    """

    @given(st.lists(raw_entries(), max_size=4))
    def test_writes_what_the_extractor_read(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        entries: list[RawJson],
    ) -> None:
        """The whole pipeline runs, from a parsed dump to a file on disk."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, entries)

        output_path = directory / "senses.jsonl"
        result = cli("collect", str(output_path), cache_dir=cache_dir)

        assert result.exit_code == 0
        assert [record["lemma"] for record in collected(output_path)] == [
            entry["word"] for entry in entries
        ]

    @given(st.lists(raw_entries(), max_size=4), st.data())
    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        entries: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """A filter named on the command line reaches the extractor."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, entries)

        allowed = data.draw(
            st.lists(parts_of_speech, min_size=1, max_size=3, unique=True)
        )
        asked = [argument for pos in allowed for argument in ("--pos", pos.value)]

        output_path = directory / "senses.jsonl"
        _ = cli("collect", str(output_path), *asked, cache_dir=cache_dir)

        codes = {pos.value for pos in allowed}

        assert [record["lemma"] for record in collected(output_path)] == [
            entry["word"] for entry in entries if entry["pos"] in codes
        ]

    @given(st.sampled_from(["--min-year", "--max-year"]), st.data())
    def test_bounds_the_quotations_the_way_round_they_were_named(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        option: str,
        data: st.DataObject,
    ) -> None:
        """The oldest and the newest reach the extractor as themselves."""
        older = data.draw(years)
        newer = data.draw(years.filter(lambda year: year > older))

        dated = [
            data.draw(raw_examples(references=references(year)))
            for year in (older, newer)
        ]
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(examples=st.just(dated)),
                    min_size=1,
                    max_size=1,
                )
            )
        )

        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, [entry])

        kept = newer if option == "--min-year" else older

        output_path = directory / "senses.jsonl"
        _ = cli("collect", str(output_path), option, str(kept), cache_dir=cache_dir)

        (record,) = collected(output_path)

        assert _years_of(record) == [kept]

    @given(st.lists(raw_entries(), max_size=3), dump_dates, st.data())
    def test_collects_the_dump_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        entries: list[RawJson],
        date: str,
        data: st.DataObject,
    ) -> None:
        """A dump named on the command line is the one read, latest or not."""
        directory = workspace()
        cache_dir = directory / "cache"

        newer = data.draw(dump_dates.filter(lambda other: other > date))
        _ = parse_dump(cache_dir, entries, date=date)
        _ = parse_dump(cache_dir, [], date=newer)

        output_path = directory / "senses.jsonl"
        _ = cli(
            "collect",
            str(output_path),
            "--dump-date",
            date,
            cache_dir=cache_dir,
        )

        assert [record["lemma"] for record in collected(output_path)] == [
            entry["word"] for entry in entries
        ]

    @given(languages, st.data())
    def test_takes_its_settings_from_the_environment(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        language: str,
        data: st.DataObject,
    ) -> None:
        """An option a whole session shares is read from the environment."""
        entries = data.draw(
            st.lists(raw_entries(languages=st.just(language)), min_size=1, max_size=3)
        )

        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, entries, language=language)

        output_path = directory / "senses.jsonl"
        result = cli(
            "collect",
            str(output_path),
            cache_dir=cache_dir,
            env={"WSC_LANGUAGE": language},
        )

        assert result.exit_code == 0
        assert [record["lemma"] for record in collected(output_path)] == [
            entry["word"] for entry in entries
        ]

    def test_refuses_when_nothing_was_parsed(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A fetched dump is not a parsed one, and the refusal says which is missing."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = fetch_dump(cache_dir)

        result = cli("collect", str(directory / "senses.jsonl"), cache_dir=cache_dir)

        assert result.exit_code != 0
        assert "parse it first" in _said(result)

    def test_refuses_when_nothing_was_fetched(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
    ) -> None:
        """An empty cache is reported rather than tripped over."""
        directory = workspace()

        result = cli(
            "collect",
            str(directory / "senses.jsonl"),
            cache_dir=directory / "cache",
        )

        assert result.exit_code != 0
        assert "fetch one first" in _said(result)

    @given(_SUFFIXES)
    def test_refuses_a_format_it_cannot_write(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        suffix: str,
    ) -> None:
        """The suffix picks the format, so an unknown one is refused."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, [])

        result = cli(
            "collect",
            str(directory / f"senses.{suffix}"),
            cache_dir=cache_dir,
        )

        assert result.exit_code != 0


class TestWordNet:
    """
    Downloading the wordnet, and settling which edition that is.
    """

    @given(wordnet_versions)
    def test_resolves_latest_against_the_index(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        edition: str,
    ) -> None:
        """Editions come out yearly, and only the index says which is the newest."""
        cache_dir = workspace() / "cache"

        with _en_word_net(edition):
            result = cli("wordnet", cache_dir=cache_dir)

        assert result.exit_code == 0
        assert f"Resolved latest to {edition}" in _said(result)
        assert cache.wordnet_path(cache_dir, edition).read_bytes() == b"a wordnet"

    @given(wordnet_versions)
    def test_fetches_the_edition_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        edition: str,
    ) -> None:
        """An edition named on the command line is fetched without asking the index."""
        cache_dir = workspace() / "cache"

        with _en_word_net(edition) as server:
            result = cli(
                "wordnet",
                "--wordnet-version",
                edition,
                cache_dir=cache_dir,
            )

            assert [call.request.url for call in server.calls] == [
                WORDNET_URL.format(version=edition)
            ]

        assert result.exit_code == 0
        assert "Resolved latest" not in _said(result)
        assert cache.wordnet_path(cache_dir, edition).read_bytes() == b"a wordnet"

    @given(wordnet_versions)
    def test_names_the_collector_and_its_version(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        edition: str,
    ) -> None:
        """A download says who answers for it wherever it is sent."""
        cache_dir = workspace() / "cache"
        expected = USER_AGENT.format(version=version("wsc"))

        with _en_word_net(edition) as server:
            _ = cli("wordnet", cache_dir=cache_dir)

            assert server.calls
            assert all(
                call.request.headers["User-Agent"] == expected for call in server.calls
            )

    @given(wordnet_versions)
    def test_stops_when_what_it_would_fetch_is_already_there(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_wordnet: Callable[..., Path],
        edition: str,
    ) -> None:
        """An edition never changes, so once it is here there is nothing to do."""
        cache_dir = workspace() / "cache"
        _ = fetch_wordnet(cache_dir, edition)

        with responses.RequestsMock() as server:
            result = cli(
                "wordnet",
                "--wordnet-version",
                edition,
                cache_dir=cache_dir,
            )

            assert not server.calls

        assert result.exit_code == 0
        assert "Already fetched" in _said(result)


class TestHelp:
    """
    What the command line says about itself.
    """

    def test_says_where_the_cache_goes_by_default(
        self,
    ) -> None:
        """None is not an answer a reader can act on, so the help says what it means."""
        result = CliRunner().invoke(app, ["fetch", "--help"])

        assert "cache directory" in _said(result)
