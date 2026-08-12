"""
Tests for src/wsc/extract/wiktionary.py.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from wsc.extract import WiktionaryExtractor
from wsc.models import POS, Example, Lemma, Quotation, Sentence

type RawJson = dict[str, object]
"""One decoded JSON object, as wiktextract writes them."""


@pytest.fixture
def extract(
    tmp_path: Path,
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
) -> Callable[..., list[Lemma]]:
    """
    Run an extraction over entries a test spells out.

    Args:
        tmp_path: The directory pytest set aside for this test.
        write_entries: Writes the entries where the extractor will read them.

    Returns:
        A runner taking the entries and the filters, and handing back the
        lemmas that came through.
    """

    def run(
        entries: Iterable[RawJson],
        language: str = "en",
        allowed_pos: frozenset[POS] | None = None,
        minimum_year: int | None = None,
        maximum_year: int | None = None,
        name: str = "wiktextract.jsonl",
    ) -> list[Lemma]:
        path = write_entries(tmp_path / name, entries)

        extractor = WiktionaryExtractor(
            language,
            allowed_pos,
            minimum_year,
            maximum_year,
        )

        return list(extractor.extract(path))

    return run


@pytest.fixture
def illustrate(
    make_entry: Callable[..., RawJson],
    make_sense: Callable[..., RawJson],
) -> Callable[..., list[RawJson]]:
    """
    Build one entry whose single sense carries the examples given.

    Args:
        make_entry: Builds the entry the sense hangs off.
        make_sense: Builds the sense the examples hang off.

    Returns:
        A builder taking raw examples and handing back entries to extract.
    """

    def build(
        *examples: RawJson,
    ) -> list[RawJson]:
        return [make_entry(senses=[make_sense(examples=list(examples))])]

    return build


@pytest.fixture
def sentences(
    extract: Callable[..., list[Lemma]],
    illustrate: Callable[..., list[RawJson]],
) -> Callable[..., list[Sentence]]:
    """
    Read back the sentences of a single sense, once the filters have run.

    Args:
        extract: Runs the extraction.
        illustrate: Builds the entry the examples hang off.

    Returns:
        A runner taking raw examples and the year bounds, and handing back
        the sentences that survive them.
    """

    def run(
        *examples: RawJson,
        minimum_year: int | None = None,
        maximum_year: int | None = None,
    ) -> list[Sentence]:
        lemmas = extract(
            illustrate(*examples),
            minimum_year=minimum_year,
            maximum_year=maximum_year,
        )

        return lemmas[0].senses[0].sentences

    return run


class TestOpening:
    """
    Reading the file however it was compressed.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "wiktextract.jsonl",
            "wiktextract.jsonl.zst",
            "wiktextract.jsonl.gz",
        ],
    )
    def test_reads_what_the_suffix_names(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        name: str,
    ) -> None:
        """A parse writes zstd, but a file found elsewhere may be plain or gzipped."""
        assert extract([make_entry()], name=name)


class TestEntries:
    """
    Which entries are read at all.
    """

    def test_keeps_the_language_asked_for(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """A dump holds every language Wiktionary describes, not the one alone."""
        entries = [
            make_entry(word="bank", lang_code="en"),
            make_entry(word="banca", lang_code="it"),
        ]

        assert [lemma.lemma for lemma in extract(entries, language="en")] == ["bank"]

    def test_reads_another_edition_the_same_way(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Nothing in the reading is English; the language is only asked for."""
        entries = [make_entry(word="banca", lang_code="it")]

        assert [lemma.lemma for lemma in extract(entries, language="it")] == ["banca"]

    def test_skips_a_part_of_speech_it_does_not_keep(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Wiktionary knows far more parts of speech than the collector keeps."""
        entries = [make_entry(pos="noun"), make_entry(word="ouch", pos="intj")]

        assert [lemma.pos for lemma in extract(entries)] == [POS.NOUN]

    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """The filter narrows what is kept, and never widens it."""
        entries = [make_entry(pos="noun"), make_entry(word="run", pos="verb")]

        lemmas = extract(entries, allowed_pos=frozenset({POS.VERB}))

        assert [lemma.lemma for lemma in lemmas] == ["run"]

    @pytest.mark.parametrize("word", ["", "   "])
    def test_skips_an_entry_with_no_headword(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        word: str,
    ) -> None:
        """A lemma is its headword, so without one there is nothing to collect."""
        assert extract([make_entry(word=word)]) == []

    def test_skips_an_entry_whose_senses_carry_no_gloss(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """An entry that defines nothing has nothing to collect either."""
        entries = [make_entry(senses=[make_sense(glosses=[])])]

        assert extract(entries) == []

    def test_skips_a_line_that_is_not_an_entry(
        self,
        tmp_path: Path,
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Wiktextract reports itself among the entries, so not every line is one."""
        path = tmp_path / "wiktextract.jsonl"
        _ = path.write_text(
            f"wiktextract is working\n{json.dumps(make_entry())}\nnot json either\n",
            encoding="utf-8",
        )

        extractor = WiktionaryExtractor("en", None, None, None)

        assert len(list(extractor.extract(path))) == 1


class TestIdentifiers:
    """
    How a lemma and its senses are named.
    """

    def test_names_a_lemma_after_its_headword_and_part_of_speech(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """An identifier reads as bank.noun.1, the ordinal opening at one."""
        assert extract([make_entry()])[0].id == "bank.noun.1"

    def test_tells_apart_entries_sharing_a_lemma_and_a_part_of_speech(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Wiktionary separates them for a reason: different etymologies."""
        entries = [make_entry(), make_entry()]

        assert [lemma.id for lemma in extract(entries)] == [
            "bank.noun.1",
            "bank.noun.2",
        ]

    def test_counts_entries_however_the_file_is_ordered(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """Entries sharing a lemma need not sit next to one another."""
        entries = [make_entry(), make_entry(word="run", pos="verb"), make_entry()]

        assert [lemma.id for lemma in extract(entries)] == [
            "bank.noun.1",
            "run.verb.1",
            "bank.noun.2",
        ]

    def test_counts_each_part_of_speech_on_its_own(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
    ) -> None:
        """The count is per lemma and part of speech, not per lemma."""
        entries = [make_entry(word="bank"), make_entry(word="bank", pos="verb")]

        assert [lemma.id for lemma in extract(entries)] == [
            "bank.noun.1",
            "bank.verb.1",
        ]

    def test_an_entry_that_is_dropped_spends_no_ordinal(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """Ordinals number the entries kept, so they run one after another."""
        entries = [make_entry(senses=[make_sense(glosses=[])]), make_entry()]

        assert [lemma.id for lemma in extract(entries)] == ["bank.noun.1"]

    def test_names_a_sense_after_its_lemma_and_its_position(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """A sense identifier reads as bank.noun.1.03, padded so that it sorts."""
        entries = [
            make_entry(
                senses=[make_sense(glosses=[f"Meaning {n}."]) for n in range(1, 4)]
            )
        ]

        assert [sense.id for sense in extract(entries)[0].senses] == [
            "bank.noun.1.01",
            "bank.noun.1.02",
            "bank.noun.1.03",
        ]

    def test_a_sense_that_is_dropped_leaves_no_gap(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """Senses are numbered as they are kept, so the count never skips."""
        entries = [
            make_entry(
                senses=[
                    make_sense(glosses=["First."]),
                    make_sense(glosses=[]),
                    make_sense(glosses=["Second."]),
                ]
            )
        ]

        assert [sense.id for sense in extract(entries)[0].senses] == [
            "bank.noun.1.01",
            "bank.noun.1.02",
        ]


class TestSenses:
    """
    What a sense carries over.
    """

    def test_keeps_the_gloss_chain_outermost_first(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """The chain is what lets a sub-sense be read on its own."""
        chain = ["A financial institution.", "Its building."]
        entries = [make_entry(senses=[make_sense(glosses=chain)])]

        assert extract(entries)[0].senses[0].glosses == tuple(chain)

    def test_keeps_a_sub_sense_alongside_its_parent(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """A parent is not replaced by what nests under it: it has examples too."""
        entries = [
            make_entry(
                senses=[
                    make_sense(glosses=["A financial institution."]),
                    make_sense(glosses=["A financial institution.", "Its building."]),
                ]
            )
        ]

        senses = extract(entries)[0].senses

        assert [sense.depth for sense in senses] == [1, 2]
        assert [sense.gloss for sense in senses] == [
            "A financial institution.",
            "Its building.",
        ]

    def test_strips_a_gloss_and_drops_a_blank_one(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """Whitespace is not a level of nesting."""
        entries = [
            make_entry(senses=[make_sense(glosses=["  A meaning.  ", "   ", "More."])])
        ]

        assert extract(entries)[0].senses[0].glosses == ("A meaning.", "More.")

    def test_keeps_tags_and_topics_apart(
        self,
        extract: Callable[..., list[Lemma]],
        make_entry: Callable[..., RawJson],
        make_sense: Callable[..., RawJson],
    ) -> None:
        """Labels of grammar are not subject fields, and neither takes the other."""
        entries = [
            make_entry(senses=[make_sense(tags=["figurative"], topics=["mathematics"])])
        ]

        sense = extract(entries)[0].senses[0]

        assert (sense.tags, sense.topics) == (("figurative",), ("mathematics",))


class TestSentences:
    """
    The sentences illustrating a sense.
    """

    def test_a_sentence_with_no_reference_is_an_example(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """An editor wrote it, so there is no source to name."""
        assert sentences(make_example("He runs.")) == [Example("He runs.")]

    def test_a_sentence_with_a_reference_is_a_quotation(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """A reference is what makes a sentence evidence from somewhere."""
        example = make_example("He runs.", ref="1999, A Book, page 1")

        assert sentences(example) == [
            Quotation("He runs.", "1999, A Book, page 1", 1999)
        ]

    def test_strips_a_sentence_and_drops_a_blank_one(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """A sentence of whitespace illustrates nothing."""
        kept = make_example("  He runs.  ")

        assert sentences(kept, make_example("   ")) == [Example("He runs.")]

    def test_keeps_the_order_they_were_listed_in(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """Wiktionary lists the plainest first, and that ordering is worth keeping."""
        listed = sentences(make_example("First."), make_example("Second."))

        assert [sentence.text for sentence in listed] == ["First.", "Second."]


class TestYears:
    """
    Reading a year off a reference, and filtering on it.
    """

    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("1999, John Smith, A Book", 1999),
            ("2026 August 11, A Newspaper", 2026),
            ("1990s, A Song", 1990),
            ("c. 1600, A Play", 1600),
            ("1999–2001, A Series", 1999),
            ("A Book, no date", None),
            ("A Book, ISBN 1234567", None),
        ],
    )
    def test_reads_the_first_year_the_reference_names(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
        reference: str,
        expected: int | None,
    ) -> None:
        """A reference is prose, so the year is taken where it is recognised."""
        quotation = sentences(make_example("He runs.", ref=reference))[0]

        assert isinstance(quotation, Quotation)
        assert quotation.year == expected

    @pytest.mark.parametrize(
        ("minimum_year", "maximum_year", "expected"),
        [
            (None, None, [1800, 1900, 2000]),
            (1900, None, [1900, 2000]),
            (None, 1900, [1800, 1900]),
            (1900, 1900, [1900]),
            (2100, None, []),
        ],
    )
    def test_keeps_the_quotations_inside_the_bounds(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
        minimum_year: int | None,
        maximum_year: int | None,
        expected: list[int],
    ) -> None:
        """Both bounds are inclusive, and either stands on its own."""
        dated = [
            make_example("He runs.", ref=f"{year}, A Book")
            for year in (1800, 1900, 2000)
        ]

        kept = sentences(
            *dated,
            minimum_year=minimum_year,
            maximum_year=maximum_year,
        )

        assert [
            sentence.year for sentence in kept if isinstance(sentence, Quotation)
        ] == expected

    def test_drops_an_undated_quotation_once_a_bound_is_set(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """A quotation that cannot be dated cannot be shown to be inside a bound."""
        undated = make_example("He runs.", ref="A Book")

        assert sentences(undated, minimum_year=1900) == []

    def test_keeps_an_undated_quotation_when_no_bound_is_set(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """An undated quotation only stands in the way once a bound is set."""
        undated = make_example("He runs.", ref="A Book")

        assert sentences(undated) == [Quotation("He runs.", "A Book", None)]

    def test_leaves_examples_alone(
        self,
        sentences: Callable[..., list[Sentence]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """The bounds reach quotations alone, an example carrying no date."""
        example = make_example("He runs.")
        quotation = make_example("She ran.", ref="1800, A Book")

        assert sentences(example, quotation, minimum_year=1900) == [Example("He runs.")]

    def test_a_sense_the_bounds_emptied_is_kept_all_the_same(
        self,
        extract: Callable[..., list[Lemma]],
        illustrate: Callable[..., list[RawJson]],
        make_example: Callable[..., RawJson],
    ) -> None:
        """A sense is what its gloss says, whatever evidence the bounds leave it."""
        entries = illustrate(make_example("He ran.", ref="1800, A Book"))

        lemmas = extract(entries, minimum_year=1900)

        assert [sense.gloss for sense in lemmas[0].senses] == ["A meaning."]
