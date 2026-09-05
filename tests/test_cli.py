"""
Tests for src/wsc/cli.py.

A command is tested for what it settles and what it refuses, not for what
the modules under it already answer for.
"""

import bz2
import gzip
import json
import string
from collections.abc import Callable, Generator
from compression import zstd
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import cast

import pytest
import responses
from documents import (
    WORDNET,
    dump,
    dump_index,
    dump_status,
    lexicon,
    wordnet_index,
)
from hypothesis import given
from hypothesis import strategies as st
from kwic import Locator
from strategies import (
    RawJson,
    dump_dates,
    parts_of_speech,
    raw_entries,
    raw_examples,
    raw_senses,
    references,
    wordnet_versions,
    years,
)
from typer.testing import CliRunner, Result

from wsc import cli as cli_module
from wsc.cli import app
from wsc.constants import (
    DUMP_INDEX_URL,
    DUMP_STATUS_URL,
    DUMP_URL,
    USER_AGENT,
    WORDNET_INDEX_URL,
    WORDNET_URL,
)
from wsc.models import Engine
from wsc.upstream import cache, wiktextract

_SYNSET_RECORD = {
    "id": "oewn-08420278-n",
    "ili": "i54321",
    "pos": "noun",
    "definition": "a financial institution.",
    "members": ["bank"],
}

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


_DUMP_BODY = bz2.compress(dump().encode())


def _gathered(
    entries: list[RawJson],
    codes: set[str] | None = None,
) -> list[object]:
    """
    Spell out the headwords a collection is expected to write.

    Entries sharing a headword and part of speech become one.

    Args:
        entries: The entries the parsed dump holds.
        codes: The parts of speech kept, or None for every one.

    Returns:
        One headword per entry gathered, in the order the first was read.
    """
    return [
        word
        for word, _ in dict.fromkeys(
            (entry["word"], entry["pos"])
            for entry in entries
            if codes is None or entry["pos"] in codes
        )
    ]


@contextmanager
def _wikimedia(
    date: str,
) -> Generator[responses.RequestsMock]:
    """
    Answer in Wikimedia's place: one edition, one dump, and it is finished.

    Args:
        date: The day the dump it holds began.

    Yields:
        The server, for the calls it took to be read back off. A command that
        was told which dump to fetch leaves the index unasked.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(DUMP_INDEX_URL, body=dump_index(date))
        _ = server.get(
            DUMP_STATUS_URL.format(date=date),
            body=dump_status("done"),
        )
        _ = server.get(DUMP_URL.format(date=date), body=_DUMP_BODY)

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
        _ = server.get(
            WORDNET_URL.format(version=version),
            body=gzip.compress(lexicon(*WORDNET).encode()),
        )

        yield server


@pytest.fixture
def cli(
    locator: Locator,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Result]:
    """
    Run a command against the cache the caller set aside.

    The search is the one exception to running a command as shipped: loading
    a real pipeline per invocation is what a suite cannot afford.

    Args:
        locator: The search every collection reads with.
        monkeypatch: Puts it in place of the one the command would load.

    Returns:
        A runner appending the cache, so that no run reads another's dumps.
    """

    def open_locator(
        _engine: Engine,
        _processes: int,
        _batch_size: int,
        *,
        gpu: bool,
    ) -> Locator:
        _ = gpu

        return locator

    monkeypatch.setattr(cli_module, "open_locator", open_locator)

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
) -> Callable[..., list[tuple[Path, Path, int, Path | None]]]:
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
    ) -> list[tuple[Path, Path, int, Path | None]]:
        calls: list[tuple[Path, Path, int, Path | None]] = []

        def parse(
            dump_path: Path,
            output_path: Path,
            processes: int,
            _narrow: object,
            database_path: Path | None,
        ) -> int:
            calls.append((dump_path, output_path, processes, database_path))

            # The command reads back what it wrote, to answer the pointers.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with zstd.open(output_path, "wt", encoding="utf-8") as file:
                _ = file.write("")

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

    @given(dump_dates)
    def test_resolves_latest_against_wikimedia(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """Latest is the newest finished dump, and only Wikimedia knows which."""
        cache_dir = workspace() / "cache"

        with _wikimedia(date):
            result = cli("fetch", cache_dir=cache_dir)

        assert result.exit_code == 0
        assert f"Resolved latest to {date}" in _said(result)
        assert (
            cache.dump_dir(cache_dir, date) / cache.DUMP_NAME
        ).read_bytes() == _DUMP_BODY

    @given(dump_dates)
    def test_fetches_the_dump_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """A dump named on the command line is fetched without asking the index."""
        cache_dir = workspace() / "cache"

        with _wikimedia(date) as server:
            result = cli(
                "fetch",
                "--dump-date",
                date,
                cache_dir=cache_dir,
            )

            assert [call.request.url for call in server.calls] == [
                DUMP_URL.format(date=date)
            ]

        assert result.exit_code == 0

    @given(dump_dates)
    def test_names_the_collector_and_its_version_to_wikimedia(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        cache_dir = workspace() / "cache"
        expected = USER_AGENT.format(version=version("wsc"))

        with _wikimedia(date) as server:
            _ = cli("fetch", cache_dir=cache_dir)

            assert server.calls
            assert all(
                call.request.headers["User-Agent"] == expected for call in server.calls
            )

    @given(dump_dates)
    def test_stops_when_what_it_would_fetch_is_already_there(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        date: str,
    ) -> None:
        """A dump is tens of gigabytes, and is not downloaded twice."""
        cache_dir = workspace() / "cache"
        _ = fetch_dump(cache_dir, date)

        with responses.RequestsMock() as server:
            result = cli(
                "fetch",
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

    @given(dump_dates, st.integers(min_value=1, max_value=16))
    def test_points_wiktextract_at_what_the_command_settled(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, int, Path | None]]],
        date: str,
        processes: int,
    ) -> None:
        """The dump to read, the file to write and the options are handed over."""
        cache_dir = workspace() / "cache"

        calls = stub_parse()
        _ = fetch_dump(cache_dir, date)

        result = cli(
            "parse",
            "--processes",
            str(processes),
            cache_dir=cache_dir,
        )

        dump_dir = cache.dump_dir(cache_dir, date)

        assert result.exit_code == 0
        assert calls == [
            (
                dump_dir / cache.DUMP_NAME,
                dump_dir / cache.WIKTEXTRACT_NAME,
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
        stub_parse: Callable[..., list[tuple[Path, Path, int, Path | None]]],
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
        assert calls[0][3] == database_path

    @given(st.lists(dump_dates, min_size=2, max_size=4, unique=True), st.data())
    def test_parses_the_dump_asked_for(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, int, Path | None]]],
        dates: list[str],
        data: st.DataObject,
    ) -> None:
        """A dump named on the command line is the one parsed, latest or not."""
        cache_dir = workspace() / "cache"

        calls = stub_parse()
        for date in dates:
            _ = fetch_dump(cache_dir, date)

        asked = data.draw(st.sampled_from(dates))
        _ = cli("parse", "--dump-date", asked, cache_dir=cache_dir)

        assert calls[0][0] == cache.dump_dir(cache_dir, asked) / cache.DUMP_NAME

    @given(st.integers(min_value=1, max_value=10000))
    def test_reports_the_lines_set_aside(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, int, Path | None]]],
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
        stub_parse: Callable[..., list[tuple[Path, Path, int, Path | None]]],
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

    def test_refuses_when_the_dump_was_not_fetched(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
    ) -> None:
        """Parsing reads a dump, so there has to be one to read."""
        result = cli(
            "parse",
            cache_dir=workspace() / "cache",
        )

        assert result.exit_code != 0
        assert "fetch one first" in _said(result)

    @given(dump_dates)
    def test_refuses_when_the_dump_is_gone_from_its_directory(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """A fetch cut short leaves the directory behind without the dump in it."""
        cache_dir = workspace() / "cache"
        cache.dump_dir(cache_dir, date).mkdir(parents=True)

        result = cli("parse", cache_dir=cache_dir)

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
        assert [record["lemma"] for record in collected(output_path)] == _gathered(
            entries
        )

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

        assert [record["lemma"] for record in collected(output_path)] == _gathered(
            entries, codes
        )

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

        assert [record["lemma"] for record in collected(output_path)] == _gathered(
            entries
        )

    @given(dump_dates, st.lists(raw_entries(), min_size=1, max_size=3))
    def test_takes_its_settings_from_the_environment(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        collected: Callable[[Path], list[RawJson]],
        date: str,
        entries: list[RawJson],
    ) -> None:
        """An option a whole session shares is read from the environment."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, entries, date)

        output_path = directory / "senses.jsonl"
        result = cli(
            "collect",
            str(output_path),
            cache_dir=cache_dir,
            env={"WSC_DUMP_DATE": date},
        )

        assert result.exit_code == 0
        assert [record["lemma"] for record in collected(output_path)] == _gathered(
            entries
        )

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


class TestWordNet:
    """
    Downloading the wordnet, reading it, and settling which edition that is.
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
        assert (cache.wordnet_dir(cache_dir, edition) / cache.WORDNET_NAME).exists()

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
                "--edition",
                edition,
                cache_dir=cache_dir,
            )

            assert [call.request.url for call in server.calls] == [
                WORDNET_URL.format(version=edition)
            ]

        assert result.exit_code == 0
        assert "Resolved latest" not in _said(result)
        assert (cache.wordnet_dir(cache_dir, edition) / cache.WORDNET_NAME).exists()

    @given(wordnet_versions)
    def test_reads_the_synsets_off_what_it_fetched(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        edition: str,
    ) -> None:
        """The archive is of no use to an alignment; the synsets in it are."""
        cache_dir = workspace() / "cache"

        with _en_word_net(edition):
            result = cli("wordnet", "--edition", edition, cache_dir=cache_dir)

        synsets_path = cache.wordnet_dir(cache_dir, edition) / cache.SYNSETS_NAME

        assert result.exit_code == 0
        assert [
            json.loads(line)
            for line in synsets_path.read_text(encoding="utf-8").splitlines()
        ] == [_SYNSET_RECORD]

    @given(wordnet_versions)
    def test_reads_a_wordnet_fetched_before(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_wordnet: Callable[..., Path],
        edition: str,
    ) -> None:
        """A fetch cut short before the read leaves the read still to do."""
        cache_dir = workspace() / "cache"
        _ = fetch_wordnet(cache_dir, edition)

        with responses.RequestsMock() as server:
            result = cli("wordnet", "--edition", edition, cache_dir=cache_dir)

            assert not server.calls

        synsets_path = cache.wordnet_dir(cache_dir, edition) / cache.SYNSETS_NAME

        assert result.exit_code == 0
        assert "Already fetched" in _said(result)
        assert [
            json.loads(line)
            for line in synsets_path.read_text(encoding="utf-8").splitlines()
        ] == [_SYNSET_RECORD]

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
    def test_stops_when_what_it_would_make_is_already_there(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        read_wordnet: Callable[..., Path],
        edition: str,
    ) -> None:
        """An edition never changes, so once it is here there is nothing to do."""
        cache_dir = workspace() / "cache"
        synsets_path = read_wordnet(cache_dir, edition, ["{}"])

        with responses.RequestsMock() as server:
            result = cli("wordnet", "--edition", edition, cache_dir=cache_dir)

            assert not server.calls

        assert result.exit_code == 0
        assert "Already read" in _said(result)
        assert synsets_path.read_text(encoding="utf-8") == "{}\n"
