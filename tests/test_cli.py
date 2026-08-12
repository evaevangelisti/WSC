"""
Tests for src/wsc/cli.py.

A command is tested for what it settles and what it refuses, not for what the
modules under it already answer for. Only collect is run end to end, since it
is the one command that reaches the disk without reaching the network.
"""

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import pytest
import responses
from typer.testing import CliRunner, Result

from wsc.cli import app
from wsc.constants import DUMP_INDEX_URL, DUMP_STATUS_URL, DUMP_URL, USER_AGENT
from wsc.dumps import cache, wiktextract

type RawJson = dict[str, object]
"""One decoded JSON object, as wiktextract writes them."""

runner = CliRunner()


@pytest.fixture
def run(
    cache_dir: Path,
) -> Callable[..., Result]:
    """
    Run a command against a cache of this test's own.

    Args:
        cache_dir: Where dumps are kept.

    Returns:
        A runner that appends the cache, so no test reads another's dumps.
    """

    def invoke(
        *arguments: str,
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
) -> Callable[..., list[tuple[Path, Path, str, int]]]:
    """
    Answer in wiktextract's place, which src/wsc/dumps/wiktextract.py is tested on.

    Args:
        monkeypatch: Puts the stand-in in place, and takes it away after.

    Returns:
        A builder taking the lines to report as set aside, and handing back
        the calls the command went on to make.
    """
    calls: list[tuple[Path, Path, str, int]] = []

    def build(
        skipped_lines: int = 0,
    ) -> list[tuple[Path, Path, str, int]]:
        def parse(
            dump_path: Path,
            output_path: Path,
            language: str,
            processes: int,
        ) -> int:
            calls.append((dump_path, output_path, language, processes))

            return skipped_lines

        monkeypatch.setattr(wiktextract, "parse", parse)

        return calls

    return build


class TestCollect:
    """
    Collecting the senses of a parsed dump into a file.
    """

    def test_writes_what_the_extractor_read(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """The whole pipeline runs, from a parsed dump to a file on disk."""
        _ = parse_dump([make_entry(word="bank"), make_entry(word="run", pos="verb")])

        output_path = tmp_path / "senses.jsonl"
        result = run("collect", str(output_path))

        assert result.exit_code == 0
        assert [
            json.loads(line)["lemma"]
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ] == ["bank", "run"]

    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """A filter named on the command line reaches the extractor."""
        _ = parse_dump([make_entry(word="bank"), make_entry(word="run", pos="verb")])

        output_path = tmp_path / "senses.jsonl"
        _ = run("collect", str(output_path), "--pos", "verb")

        assert [
            json.loads(line)["lemma"]
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ] == ["run"]

    def test_keeps_every_part_of_speech_when_none_is_named(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Naming no part of speech keeps them all, rather than none."""
        _ = parse_dump([make_entry(word="bank"), make_entry(word="run", pos="verb")])

        output_path = tmp_path / "senses.jsonl"
        _ = run("collect", str(output_path))

        assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2

    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("--min-year", 2000),
            ("--max-year", 1800),
        ],
    )
    def test_bounds_the_quotations_the_way_round_they_were_named(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
        make_example: Callable[..., RawJson],
        option: str,
        expected: int,
    ) -> None:
        """The oldest and the newest reach the extractor as themselves."""
        dated = [
            make_example("He runs.", ref=f"{year}, A Book") for year in (1800, 2000)
        ]
        _ = parse_dump([make_entry(senses=[make_sense(examples=dated)])])

        output_path = tmp_path / "senses.jsonl"
        _ = run("collect", str(output_path), option, "1900")

        assert json.loads(output_path.read_text(encoding="utf-8"))["senses"][0][
            "sentences"
        ] == [
            {
                "text": "He runs.",
                "reference": f"{expected}, A Book",
                "year": expected,
            }
        ]

    def test_collects_the_dump_asked_for(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """A dump named on the command line is the one read, latest or not."""
        _ = parse_dump([make_entry(word="bank")], date="20260701")
        _ = parse_dump([make_entry(word="run", pos="verb")], date="20260801")

        output_path = tmp_path / "senses.jsonl"
        _ = run("collect", str(output_path), "--dump-date", "20260701")

        assert json.loads(output_path.read_text(encoding="utf-8"))["lemma"] == "bank"

    def test_takes_its_settings_from_the_environment(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """An option a whole session shares is read from the environment."""
        _ = parse_dump([make_entry(word="banca", lang_code="it")], language="it")

        output_path = tmp_path / "senses.jsonl"
        _ = run("collect", str(output_path), env={"WSC_LANGUAGE": "it"})

        assert "banca" in output_path.read_text(encoding="utf-8")

    def test_refuses_when_nothing_was_parsed(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A fetched dump is not a parsed one, and the refusal says which is missing."""
        _ = fetch_dump()

        result = run("collect", str(tmp_path / "senses.jsonl"))

        assert result.exit_code != 0
        assert "parse it first" in result.output

    def test_refuses_when_nothing_was_fetched(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
    ) -> None:
        """An empty cache is reported rather than tripped over."""
        result = run("collect", str(tmp_path / "senses.jsonl"))

        assert result.exit_code != 0
        assert "fetch one first" in result.output

    def test_refuses_a_format_it_cannot_write(
        self,
        tmp_path: Path,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """The suffix picks the format, so an unknown one is refused."""
        _ = parse_dump([make_entry()])

        result = run("collect", str(tmp_path / "senses.parquet"))

        assert result.exit_code != 0


class TestParse:
    """
    What a parse settles, wiktextract standing in for itself.
    """

    def test_points_wiktextract_at_what_the_command_settled(
        self,
        cache_dir: Path,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int]]],
    ) -> None:
        """The dump to read, the file to write and the options are handed over."""
        calls = stub_parse()

        _ = fetch_dump("en", "20260801")

        result = run("parse", "--processes", "4")

        dump_dir = cache.dump_dir(cache_dir, "en", "20260801")

        assert result.exit_code == 0
        assert calls == [
            (
                dump_dir / cache.DUMP_NAME,
                dump_dir / cache.WIKTEXTRACT_NAME,
                "en",
                4,
            )
        ]
        assert "Parsed" in result.output

    def test_parses_the_dump_asked_for(
        self,
        cache_dir: Path,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int]]],
    ) -> None:
        """A dump named on the command line is the one parsed, latest or not."""
        calls = stub_parse()

        _ = fetch_dump("en", "20260701")
        _ = fetch_dump("en", "20260801")

        _ = run("parse", "--dump-date", "20260701")

        assert calls[0][0] == (
            cache.dump_dir(cache_dir, "en", "20260701") / cache.DUMP_NAME
        )

    def test_reports_the_lines_set_aside(
        self,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int]]],
    ) -> None:
        """Far more than a few hundred means something went wrong, so it is said."""
        _ = stub_parse(skipped_lines=3)
        _ = fetch_dump()

        result = run("parse")

        assert "Set aside 3 lines" in result.output

    def test_says_nothing_when_no_line_was_set_aside(
        self,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
        stub_parse: Callable[..., list[tuple[Path, Path, str, int]]],
    ) -> None:
        """A parse with nothing to report reports nothing."""
        _ = stub_parse()
        _ = fetch_dump()

        result = run("parse")

        assert "Set aside" not in result.output

    def test_refuses_when_the_dump_was_not_fetched(
        self,
        run: Callable[..., Result],
    ) -> None:
        """Parsing reads a dump, so there has to be one to read."""
        result = run("parse")

        assert result.exit_code != 0
        assert "fetch one first" in result.output

    def test_refuses_when_the_dump_is_gone_from_its_directory(
        self,
        cache_dir: Path,
        run: Callable[..., Result],
    ) -> None:
        """A fetch cut short leaves the directory behind without the dump in it."""
        cache.dump_dir(cache_dir, "en", "20260801").mkdir(parents=True)

        result = run("parse")

        assert result.exit_code != 0
        assert "fetch it first" in result.output

    def test_stops_when_the_dump_was_already_parsed(
        self,
        run: Callable[..., Result],
        parse_dump: Callable[..., Path],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """A parse takes the better part of a day, and is not repeated for nothing."""
        _ = parse_dump([make_entry()])

        result = run("parse")

        assert result.exit_code == 0
        assert "Already parsed" in result.output


class TestFetch:
    """
    Downloading a dump, and settling which one that is.
    """

    def test_stops_when_the_dump_is_already_there(
        self,
        run: Callable[..., Result],
        fetch_dump: Callable[..., Path],
    ) -> None:
        """A dump is tens of gigabytes, and is not downloaded twice."""
        _ = fetch_dump("en", "20260801")

        result = run("fetch", "--dump-date", "20260801")

        assert result.exit_code == 0
        assert "Already fetched" in result.output

    @responses.activate
    def test_resolves_latest_against_wikimedia(
        self,
        cache_dir: Path,
        run: Callable[..., Result],
    ) -> None:
        """Latest is the newest finished dump, and only Wikimedia knows which."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body='<a href="20260801/">20260801/</a>',
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=json.dumps(
                {"jobs": {"articlesdumprecombine": {"status": "done"}}},
            ),
        )
        _ = responses.get(
            DUMP_URL.format(language="en", date="20260801"),
            body=b"a dump",
        )

        result = run("fetch")

        assert result.exit_code == 0
        assert (
            cache.dump_dir(cache_dir, "en", "20260801") / cache.DUMP_NAME
        ).read_bytes() == b"a dump"

    @responses.activate
    def test_names_the_collector_and_its_version_to_wikimedia(
        self,
        run: Callable[..., Result],
    ) -> None:
        """Wikimedia asks that requests name whoever answers for them."""
        _ = responses.get(
            DUMP_INDEX_URL.format(language="en"),
            body='<a href="20260801/">20260801/</a>',
        )
        _ = responses.get(
            DUMP_STATUS_URL.format(language="en", date="20260801"),
            body=json.dumps(
                {"jobs": {"articlesdumprecombine": {"status": "done"}}},
            ),
        )
        _ = responses.get(
            DUMP_URL.format(language="en", date="20260801"),
            body=b"a dump",
        )

        _ = run("fetch")

        expected = USER_AGENT.format(version=version("wsc"))

        assert responses.calls
        assert all(
            call.request.headers["User-Agent"] == expected for call in responses.calls
        )


class TestHelp:
    """
    What the command line says about itself.
    """

    @pytest.mark.parametrize("command", ["fetch", "parse", "collect"])
    def test_every_command_documents_itself(
        self,
        command: str,
    ) -> None:
        """Every command answers for itself when its help is asked for."""
        result = runner.invoke(app, [command, "--help"])

        assert result.exit_code == 0

    def test_says_where_the_cache_goes_by_default(
        self,
    ) -> None:
        """None is not an answer a reader can act on, so the help says what it means."""
        result = runner.invoke(app, ["fetch", "--help"])

        assert "cache directory" in result.output
