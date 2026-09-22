"""
Tests for src/wsc/cli.py.

Tests exercise command behavior through its public interface.
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
from documents import dump, dump_index, dump_status, page
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
    words,
    years,
)
from typer.testing import CliRunner, Result

from wsc import cli as cli_module
from wsc.cli import app
from wsc.constants import (
    DUMP_INDEX_URL,
    DUMP_STATUS_URL,
    DUMP_URL,
    KAIKKI_URL,
    USER_AGENT,
)
from wsc.models import Engine
from wsc.reading import read_lemmas
from wsc.upstream import cache, wiktextract


def _normalize_output(
    result: Result,
) -> str:
    """
    Read what a command said, with the wrapping taken out.

    Args:
        result: What the run handed back.

    Returns:
        Command output normalized to one line.
    """
    return " ".join(result.output.replace("│", " ").split())


def _read_years(
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


def _collect_headwords(
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
def _serve_wikimedia(
    date: str,
) -> Generator[responses.RequestsMock]:
    """
    Answer in Wikimedia's place: one edition, one dump, and it is finished.

    Args:
        date: The day the dump it holds began.

    Yields:
        A server fixture recording requests for the selected dump.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as server:
        _ = server.get(DUMP_INDEX_URL, body=dump_index(date))
        _ = server.get(
            DUMP_STATUS_URL.format(date=date),
            body=dump_status("done"),
        )
        _ = server.get(DUMP_URL.format(date=date), body=_DUMP_BODY)

        yield server


@pytest.fixture
def cli(
    locator: Locator,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Result]:
    """
    Run a command against the cache the caller set aside.

    The search is the one exception to running a command as shipped: loading a real
    pipeline per invocation is what a suite cannot afford.

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
        """
        Supply the deterministic locator used by command tests.

        Args:
            _engine: Requested engine, replaced by the deterministic test engine.
            _processes: Requested worker count, unused by the test engine.
            _batch_size: Requested batch size, unused by the test engine.
            gpu: Requested GPU setting, unused by the test engine.

        Returns:
            The shared locator without loading a model.
        """
        _ = gpu

        return locator

    monkeypatch.setattr(cli_module, "open_locator", open_locator)

    runner = CliRunner()

    def invoke(
        *arguments: str,
        cache_dir: Path,
        env: dict[str, str] | None = None,
    ) -> Result:
        """
        Run a command with an isolated source cache.

        Args:
            arguments: Command name and command-line arguments.
            cache_dir: Isolated source cache directory.
            env: Environment overrides passed to the command runner.

        Returns:
            The command output and exit status.
        """
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
        A builder recording parse calls and discarded reporting lines.
    """

    def build(
        skipped_lines: int = 0,
    ) -> list[tuple[Path, Path, int, Path | None]]:
        """
        Install a parser replacement recording its arguments.

        Args:
            skipped_lines: Number of reporting lines the parser should discard.

        Returns:
            The list populated by subsequent parser calls.
        """
        calls: list[tuple[Path, Path, int, Path | None]] = []

        def parse(
            dump_path: Path,
            output_path: Path,
            processes: int,
            _narrow: object,
            database_path: Path | None,
        ) -> int:
            """
            Record a parser invocation and create empty compressed output.

            Args:
                dump_path: Cached dump passed to the parser.
                output_path: Destination of the generated extraction.
                processes: Requested parser worker count.
                _narrow: Narrowing callback unused by the parser replacement.
                database_path: Requested Wiktextract page database, or None.

            Returns:
                The configured number of discarded reporting lines.
            """
            calls.append((dump_path, output_path, processes, database_path))

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
        A reader decoding each JSONL record.
    """

    def read(
        output_path: Path,
    ) -> list[RawJson]:
        """
        Decode the entries written by a collection command.

        Args:
            output_path: Destination of the generated extraction.

        Returns:
            One JSON object per collected entry.
        """
        text = output_path.read_text(encoding="utf-8")

        return [json.loads(line) for line in text.split("\n") if line]

    return read


class TestFetch:
    """Downloading a dump, and settling which one that is."""

    @given(dump_dates)
    def test_resolves_latest_dump(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """Latest is the newest finished dump, and only Wikimedia knows which."""
        cache_dir = workspace() / "cache"

        with _serve_wikimedia(date):
            result = cli("fetch", cache_dir=cache_dir)

        assert result.exit_code == 0
        assert f"Resolved latest to {date}" in _normalize_output(result)
        assert (
            cache.dump_dir(cache_dir, date) / cache.DUMP_NAME
        ).read_bytes() == _DUMP_BODY

    @given(dump_dates)
    def test_fetches_selected_dump(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """A dump named on the command line is fetched without asking the index."""
        cache_dir = workspace() / "cache"

        with _serve_wikimedia(date) as server:
            result = cli(
                "fetch",
                "--dump-date",
                date,
                cache_dir=cache_dir,
            )

            assert [call.request.url for call in server.calls] == [
                DUMP_URL.format(date=date),
            ]

        assert result.exit_code == 0

    @given(dump_dates)
    def test_sends_package_identity(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        date: str,
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        cache_dir = workspace() / "cache"
        expected = USER_AGENT.format(version=version("wsc"))

        with _serve_wikimedia(date) as server:
            _ = cli("fetch", cache_dir=cache_dir)

            assert server.calls
            assert all(
                call.request.headers["User-Agent"] == expected for call in server.calls
            )

    @given(dump_dates)
    def test_reuses_cached_dump(
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
        assert "Already fetched" in _normalize_output(result)


class TestParse:
    """What a parse settles, wiktextract standing in for itself."""

    @given(dump_dates, st.integers(min_value=1, max_value=16))
    def test_forwards_parser_settings(
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
            ),
        ]
        assert "Parsed" in _normalize_output(result)

    @given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=8))
    def test_forwards_database_path(
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
    def test_parses_selected_dump(
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
    def test_reports_discarded_lines(
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

        assert f"WARNING wsc.cli: Skipped {skipped_lines} lines" in _normalize_output(
            result,
        )

    def test_omits_empty_warnings(
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

        assert "Set aside" not in _normalize_output(result)

    @given(st.lists(raw_entries(), max_size=3))
    def test_reuses_parsed_dump(
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
        assert "Already parsed" in _normalize_output(result)

    def test_rejects_unfetched_dump(
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
        assert "fetch one first" in _normalize_output(result)

    @given(dump_dates)
    def test_rejects_missing_dump(
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
        assert "fetch it first" in _normalize_output(result)


@pytest.mark.parametrize("cached", [False, True])
def test_collects_published_archive(
    tmp_path: Path,
    cli: Callable[..., Result],
    fetch_dump: Callable[..., Path],
    *,
    cached: bool,
) -> None:
    """Archive collection retains lexical evidence and supplemental translations."""
    dump_path = fetch_dump(
        tmp_path,
        pages=[
            page(
                "Rome",
                "==English==\n===Proper noun===\n{{trans-top|capital}}\n"
                + "{{t|it|Roma}}\n{{trans-bottom}}",
            ),
        ],
    )
    records = [
        {
            "word": "Rome",
            "pos": "name",
            "lang_code": "en",
            "unused": "discard",
            "senses": [
                {
                    "glosses": ["A capital city."],
                    "wikidata": ["Q220"],
                    "examples": [
                        {
                            "text": "Rome",
                            "type": "example",
                            "bold_text_offsets": [[0, 4]],
                        },
                    ],
                },
            ],
        },
        {
            "word": "R.",
            "pos": "name",
            "lang_code": "en",
            "senses": [{"tags": ["alt-of"], "alt_of": [{"word": "Rome"}]}],
        },
    ]
    archive_bytes = gzip.compress(
        "".join(f"{json.dumps(record)}\n" for record in records).encode(),
    )
    archive_path = dump_path.with_name(cache.ARCHIVE_NAME)

    if cached:
        _ = archive_path.write_bytes(archive_bytes)

    with responses.RequestsMock() as server:
        if not cached:
            _ = server.get(KAIKKI_URL, body=archive_bytes)

        parsed = cli("parse", "--archive", cache_dir=tmp_path)

        assert parsed.exit_code == 0, parsed.output
        assert len(server.calls) == (0 if cached else 1)

    output_dir = tmp_path / "collection"
    collected = cli(
        "collect",
        "--engine",
        "lemminflect",
        "--output-dir",
        str(output_dir),
        cache_dir=tmp_path,
    )

    assert collected.exit_code == 0, collected.output
    assert archive_path.read_bytes() == archive_bytes

    (entry,) = read_lemmas(output_dir / "senses.jsonl")

    assert entry.id == "Rome.propn"
    assert entry.variants == frozenset({"R."})
    assert entry.senses[0].wikidata_ids == ("Q220",)

    (offset,) = entry.senses[0].sentences[0].word_offsets

    assert offset.offset == (0, 4)
    assert offset.sources == ("bold", "lemmatizer")
    assert entry.translation_tables[0].translations == {"it": frozenset({"Roma"})}


class TestCollect:
    """Collecting the senses of a parsed dump into an output directory."""

    def test_writes_default_directory(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        parse_dump: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An omitted output option creates all artifacts in collection."""
        directory = workspace()
        cache_dir = directory / "cache"
        input_path = parse_dump(cache_dir, [])
        monkeypatch.chdir(directory)

        result = cli("collect", cache_dir=cache_dir)

        assert result.exit_code == 0

        output_dir = directory / "collection"
        manifest = cast(
            RawJson,
            json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")),
        )
        sources = cast(dict[str, RawJson], manifest["sources"])
        settings = cast(RawJson, manifest["settings"])

        assert {path.name for path in output_dir.iterdir()} == {
            "senses.jsonl",
            "report.json",
            "report.md",
            "manifest.json",
        }
        assert manifest["dump_date"] == "20260801"
        assert sources["wiktextract"]["path"] == str(input_path.resolve())
        assert sources["wiktextract"]["bytes"] == input_path.stat().st_size
        assert settings["engine"] == "spacy"

    @given(st.lists(raw_entries(), max_size=4))
    def test_exports_collected_entries(
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

        output_dir = directory / "collection"
        output_path = output_dir / "senses.jsonl"
        result = cli("collect", "--output-dir", str(output_dir), cache_dir=cache_dir)

        assert result.exit_code == 0
        assert [
            record["lemma"] for record in collected(output_path)
        ] == _collect_headwords(entries)

    @given(st.lists(raw_entries(), max_size=4), st.data())
    def test_filters_selected_categories(
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
            st.lists(parts_of_speech, min_size=1, max_size=3, unique=True),
        )
        asked = [argument for pos in allowed for argument in ("--pos", pos.value)]

        output_dir = directory / "collection"
        output_path = output_dir / "senses.jsonl"
        _ = cli("collect", "--output-dir", str(output_dir), *asked, cache_dir=cache_dir)

        codes = {pos.value for pos in allowed}

        assert [
            record["lemma"] for record in collected(output_path)
        ] == _collect_headwords(entries, codes)

        report = cast(
            RawJson,
            json.loads((output_dir / "report.json").read_text(encoding="utf-8")),
        )
        manifest = cast(
            RawJson,
            json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")),
        )
        settings = cast(RawJson, manifest["settings"])

        assert set(cast(dict[str, int], report["entries"])) <= codes
        assert settings["parts_of_speech"] == [part.value for part in allowed]

    @given(st.sampled_from(["--min-year", "--max-year"]), st.data())
    def test_orders_quotation_bounds(
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
            data.draw(
                raw_examples(
                    texts=words,
                    references=references(year),
                ),
            )
            for year in (older, newer)
        ]
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(examples=st.just(dated)),
                    min_size=1,
                    max_size=1,
                ),
            ),
        )

        directory = workspace()
        cache_dir = directory / "cache"
        _ = parse_dump(cache_dir, [entry])

        kept = newer if option == "--min-year" else older

        output_dir = directory / "collection"
        output_path = output_dir / "senses.jsonl"
        _ = cli(
            "collect",
            "--output-dir",
            str(output_dir),
            option,
            str(kept),
            cache_dir=cache_dir,
        )

        (record,) = collected(output_path)

        assert _read_years(record) == [kept]

    @given(st.lists(raw_entries(), max_size=3), dump_dates, st.data())
    def test_collects_selected_dump(
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

        output_dir = directory / "collection"
        output_path = output_dir / "senses.jsonl"
        _ = cli(
            "collect",
            "--output-dir",
            str(output_dir),
            "--dump-date",
            date,
            cache_dir=cache_dir,
        )

        assert [
            record["lemma"] for record in collected(output_path)
        ] == _collect_headwords(entries)

        manifest = cast(
            RawJson,
            json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")),
        )

        assert manifest["dump_date"] == date

    @given(dump_dates, st.lists(raw_entries(), min_size=1, max_size=3))
    def test_reads_environment_settings(
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

        output_dir = directory / "collection"
        output_path = output_dir / "senses.jsonl"
        result = cli(
            "collect",
            "--output-dir",
            str(output_dir),
            cache_dir=cache_dir,
            env={"WSC_DUMP_DATE": date},
        )

        assert result.exit_code == 0
        assert [
            record["lemma"] for record in collected(output_path)
        ] == _collect_headwords(entries)

    def test_rejects_missing_extraction(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A fetched dump is not a parsed one, and the refusal says which is missing."""
        directory = workspace()
        cache_dir = directory / "cache"
        _ = fetch_dump(cache_dir)

        result = cli(
            "collect",
            "--output-dir",
            str(directory / "collection"),
            cache_dir=cache_dir,
        )

        assert result.exit_code != 0
        assert "parse it first" in _normalize_output(result)

    def test_rejects_empty_cache(
        self,
        workspace: Callable[[], Path],
        cli: Callable[..., Result],
    ) -> None:
        """An empty cache produces an explicit error."""
        directory = workspace()

        result = cli(
            "collect",
            "--output-dir",
            str(directory / "collection"),
            cache_dir=directory / "cache",
        )

        assert result.exit_code != 0
        assert "fetch one first" in _normalize_output(result)


class TestHelp:
    """What the command line says about itself."""

    def test_describes_default_cache(
        self,
    ) -> None:
        """None is not an answer a reader can act on, so the help says what it means."""
        result = CliRunner().invoke(app, ["fetch", "--help"])

        assert "cache directory" in _normalize_output(result)

    def test_describes_alignment_verbosity(
        self,
    ) -> None:
        """Alignment help exposes the opt-in detailed logging switch."""
        result = CliRunner().invoke(app, ["align", "--help"])

        assert "--verbose" in _normalize_output(result)
