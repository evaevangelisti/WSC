"""Exercise complete collection artifacts through the public writing API."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from statistics import median
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import RawJson, lemmas

from wsc.collection import write_collection
from wsc.models import (
    POS,
    Example,
    Lemma,
    Quotation,
    Sense,
    TranslationTable,
    WordOffset,
    WordOffsetSource,
)


def _read_document(
    path: Path,
) -> RawJson:
    """Read a generated JSON document without discarding numeric values."""
    return cast(RawJson, json.loads(path.read_text(encoding="utf-8")))


@given(st.lists(lemmas, max_size=6))
def test_preserves_collection_totals(
    workspace: Callable[[], Path],
    entries: list[Lemma],
) -> None:
    """One-pass output accounts for every entry and its evidence."""
    output_dir = workspace() / "collection"

    write_collection(iter(entries), output_dir, {"dump_date": "20260801"})

    with (output_dir / "senses.jsonl").open(encoding="utf-8") as stream:
        records = [cast(RawJson, json.loads(line)) for line in stream]

    assert [(record["id"], record["lemma"], record["pos"]) for record in records] == [
        (entry.id, entry.lemma, entry.pos) for entry in entries
    ]
    assert {path.name for path in output_dir.iterdir()} == {
        "senses.jsonl",
        "report.json",
        "report.md",
        "manifest.json",
    }

    report = _read_document(output_dir / "report.json")
    manifest = _read_document(output_dir / "manifest.json")
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    senses = [sense for entry in entries for sense in entry.senses]
    sentences = [sentence for sense in senses for sentence in sense.sentences]
    offsets = [offset for sentence in sentences for offset in sentence.word_offsets]
    totals = cast(RawJson, report["totals"])
    written_senses = [
        sense
        for record in records
        for sense in cast(list[RawJson], record.get("senses", []))
    ]

    assert [sense["id"] for sense in written_senses] == [sense.id for sense in senses]

    assert totals["entries"] == len(entries)
    assert totals["senses"] == len(senses)
    assert totals["sentences"] == len(sentences)
    assert totals["offsets"] == len(offsets)
    assert manifest["totals"] == totals
    assert manifest["dump_date"] == "20260801"
    assert report["unattested_senses"] == sum(not sense.sentences for sense in senses)

    occurrences = cast(dict[str, int], report["offsets_per_sentence"])
    identifiers = cast(dict[str, int], report["wikidata_ids_per_sense"])
    languages = cast(dict[str, int], report["translation_languages"])

    assert sum(occurrences.values()) == len(sentences)
    assert sum(int(number) * count for number, count in occurrences.items()) == len(
        offsets
    )
    assert sum(identifiers.values()) == len(senses)
    assert sum(languages.values()) == report["translations"]

    summary = markdown.split("### Senses per entry\n", 1)[1].split("\n###", 1)[0]

    assert "| Statistic | Value |" in summary

    if entries:
        expected = median(len(entry.senses) for entry in entries)

        assert f"| Median | {float(expected)} |" in summary
    else:
        assert "| Median | — |" in summary

    dated = [
        sentence.year
        for sentence in sentences
        if isinstance(sentence, Quotation) and sentence.year is not None
    ]

    if dated:
        quotation_summary = markdown.split("### Quotation years\n", 1)[1].split(
            "\n##", 1
        )[0]

        assert f"| Median | {round(median(dated))} |" in quotation_summary


def test_reports_lexical_evidence(
    tmp_path: Path,
) -> None:
    """Reports distinguish ambiguous IDs, repeated translations, and offset sources."""
    entry = Lemma(
        "bank.noun",
        "bank",
        POS.NOUN,
        variants=frozenset({"banke"}),
        senses=[
            Sense(
                "bank.noun.1",
                ("an institution",),
                synonyms=("depository",),
                tags=("rare|historical",),
                wikidata_ids=("Q1", "Q1"),
                sentences=[
                    Quotation(
                        "bank",
                        "A reference",
                        2000,
                        word_offsets=(
                            WordOffset(
                                (0, 4),
                                (WordOffsetSource.LEMMATIZER, WordOffsetSource.BOLD),
                            ),
                        ),
                    ),
                    Quotation("BANK", "Undated reference"),
                    Example("banks"),
                ],
            ),
            Sense("bank.noun.2", ("a slope", "a riverbank"), wikidata_ids=("Q2", "Q3")),
            Sense("bank.noun.3", ("a collection",)),
        ],
        translation_tables=(
            TranslationTable(
                "bank.noun.tr.1", "institution", {"fr": frozenset({"banque"})}
            ),
            TranslationTable(
                "bank.noun.tr.2",
                "institution",
                {"fr": frozenset({"banque"}), "it": frozenset({"banca"})},
            ),
        ),
    )

    write_collection([entry], tmp_path, {})

    report = _read_document(tmp_path / "report.json")
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert report["wikidata_ids_per_sense"] == {"0": 1, "1": 1, "2": 1}
    assert report["gloss_depths"] == {"1": 2, "2": 1}
    assert report["translation_tables"] == 2
    assert report["translation_languages"] == {"fr": 2, "it": 1}
    assert report["translations"] == 3
    assert report["variants"] == 1
    assert report["synonyms"] == 1
    assert report["quotation_years"] == {"2000": 1}
    assert report["undated_quotations"] == 1
    assert report["literal_misses"] == 1
    assert report["offset_sources"] == {"bold+lemmatizer": 1}
    assert report["source_relations"] == {"Exact agreement": 1}
    assert report["offset_violations"] == {}
    assert r"rare\|historical" in markdown
    assert "| One | 1 | 33.3% |" in markdown
    assert "| Multiple | 1 | 33.3% |" in markdown
    assert markdown.startswith("# Collection report\n\n## Records\n")

    for title in ("Variants and synonyms", "Translations"):
        section = markdown.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]
        totals, averages = section.split("### Averages\n", 1)

        assert "### Totals\n\n| Figure | Count |" in totals
        assert "| Figure | Average |" in averages
        assert "per entry" not in totals
        assert "per sense" not in totals
        assert "per translated entry" not in totals


def test_reports_offset_violations(
    tmp_path: Path,
) -> None:
    """Overlapping source proposals remain distinct from malformed offsets."""
    entry = Lemma(
        "cat.noun",
        "cat",
        POS.NOUN,
        senses=[
            Sense(
                "cat.noun.1",
                ("an animal",),
                sentences=[
                    Example(
                        "cats",
                        word_offsets=(
                            WordOffset((0, 3), (WordOffsetSource.BOLD,)),
                            WordOffset((0, 4), (WordOffsetSource.LEMMATIZER,)),
                        ),
                    ),
                    Example(
                        "cat",
                        word_offsets=(WordOffset((-1, 5), (WordOffsetSource.BOLD,)),),
                    ),
                ],
            ),
        ],
    )

    write_collection([entry], tmp_path, {})

    report = _read_document(tmp_path / "report.json")

    assert report["offsets"] == 3
    assert report["source_relations"] == {"Overlapping proposals": 1, "Bold only": 1}
    assert report["offset_violations"] == {
        "Inside a longer word (bold)": 1,
        "Out of range (bold)": 1,
    }
    assert report["different_surfaces"] == 2


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["extraction", "serialization"])
def test_preserves_failed_collection(
    tmp_path: Path,
    *,
    existing: bool,
    failure: str,
) -> None:
    """Failures publish no partial reports and preserve any previous collection."""
    output_dir = tmp_path / "collection"
    entry = Lemma("bank.noun", "bank", POS.NOUN)
    expected: dict[str, bytes] = {}

    if existing:
        write_collection([entry], output_dir, {"dump_date": "20260801"})
        expected = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    def interrupted_entries() -> Iterator[Lemma]:
        """
        Yield one entry before interrupting extraction.

        Yields:
            The entry written before the simulated failure.

        Raises:
            RuntimeError: After the first entry has been consumed.
        """
        yield entry

        raise RuntimeError("Interrupted extraction")

    if failure == "extraction":
        with pytest.raises(RuntimeError, match="Interrupted extraction"):
            write_collection(interrupted_entries(), output_dir, {})
    else:
        with pytest.raises(TypeError, match="not JSON serializable"):
            write_collection([entry], output_dir, {"unsupported": object()})

    if existing:
        assert {
            path.name: path.read_bytes() for path in output_dir.iterdir()
        } == expected
    else:
        assert not output_dir.exists()

    assert not list(tmp_path.glob(".collection-*"))


def test_replaces_complete_collection(
    tmp_path: Path,
) -> None:
    """An empty rerun replaces stale records and reports from the previous run."""
    write_collection([Lemma("bank.noun", "bank", POS.NOUN)], tmp_path, {})
    write_collection([], tmp_path, {"dump_date": "20260901"})

    assert (tmp_path / "senses.jsonl").read_bytes() == b""
    assert (
        cast(RawJson, _read_document(tmp_path / "report.json")["totals"])["entries"]
        == 0
    )
    assert _read_document(tmp_path / "manifest.json")["dump_date"] == "20260901"
