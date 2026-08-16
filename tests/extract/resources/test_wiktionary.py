"""
Tests for src/wsc/extract/resources/wiktionary.py.
"""

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import (
    RawJson,
    blanks,
    glosses,
    languages,
    parts_of_speech,
    raw_entries,
    raw_examples,
    raw_forms,
    raw_senses,
    references,
    texts,
    undated_references,
    unknown_pos_codes,
    words,
    years,
)

from wsc.extract import WiktionaryExtractor
from wsc.models import POS, Example, Lemma, Quotation, Sentence

# json.loads is typed loosely; whether a line decodes at all is the whole of
# what is asked of it here.
_loads: Callable[[str], object] = json.loads


def _reads_as_an_entry(
    line: str,
) -> bool:
    """
    Say whether a line is an entry, which is a JSON object written whole.

    Args:
        line: One line of a wiktextract file.

    Returns:
        Whether it opens an object, and whether that object is finished.
    """
    if not line.startswith("{"):
        return False

    try:
        _ = _loads(line)
    except ValueError:
        return False

    return True


def _padded(
    word: str,
) -> st.SearchStrategy[str]:
    """
    Draw a word as an editor may have left it, with whitespace around it.

    Args:
        word: The word itself.

    Returns:
        A strategy over the ways it may have been written down.
    """
    return st.tuples(_PADDING, st.just(word), _PADDING).map("".join)


_PADDED_HEADWORDS = words.flatmap(_padded)


# The three ways an entry may fail to be one: a headword that is whitespace,
# a part of speech the collector does not keep, and senses defining nothing.
_UNREADABLE = st.one_of(
    raw_entries(headwords=st.just("") | blanks),
    raw_entries(pos_codes=unknown_pos_codes),
    raw_entries(
        senses=st.lists(raw_senses(glosses=st.lists(blanks, max_size=2)), max_size=2)
    ),
)

# What a wiktextract file holds beside the entries: the prose wiktextract
# writes about itself, JSON that decodes into anything but an entry, and an
# entry a dump cut short left unfinished.
_REPORTS = st.one_of(
    st.text(
        alphabet=st.characters(codec="utf-8", exclude_characters="\n\r"),
        max_size=30,
    ),
    st.sampled_from(
        [
            "0",
            "123",
            '"parsing pages"',
            "null",
            "true",
            "[]",
            '[{"word": "bank"}]',
            "{",
            '{"word": "bank"',
            '{"word": }',
        ]
    ),
).filter(lambda line: not _reads_as_an_entry(line))

# Rows describing an inflection table rather than the lemma, and the
# transliterations standing beside a form rather than for it.
_SERVICE_TAGS = st.sampled_from(["inflection-template", "romanization", "table-tags"])

# What an inflection table writes for a cell it leaves empty.
_EMPTY_CELLS = st.sampled_from(["-", ""]) | blanks

_PADDING = blanks | st.just("")


@pytest.fixture
def extract(
    workspace: Callable[[], Path],
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
) -> Callable[..., list[Lemma]]:
    """
    Run an extraction over entries a property drew.

    Args:
        workspace: Sets aside a directory for the file being read.
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
        path = write_entries(workspace() / name, entries)

        extractor = WiktionaryExtractor(
            language,
            allowed_pos,
            minimum_year,
            maximum_year,
        )

        return list(extractor.extract(path))

    return run


@pytest.fixture
def extract_lines(
    workspace: Callable[[], Path],
) -> Callable[[Iterable[str]], list[Lemma]]:
    """
    Run an extraction over the lines of a file, entries or otherwise.

    Args:
        workspace: Sets aside a directory for the file being read.

    Returns:
        A runner taking the lines and handing back the lemmas read out of
        those that were entries.
    """

    def run(
        lines: Iterable[str],
    ) -> list[Lemma]:
        path = workspace() / "wiktextract.jsonl"
        _ = path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

        return list(WiktionaryExtractor("en", None, None, None).extract(path))

    return run


@pytest.fixture
def attest(
    extract: Callable[..., list[Lemma]],
) -> Callable[..., list[Sentence]]:
    """
    Read back the sentences of a single sense, once the filters have run.

    Args:
        extract: Runs the extraction.

    Returns:
        A runner taking raw examples, the headword they attest, its forms and
        the year bounds, and handing back the sentences that survive them.
    """

    def run(
        *examples: RawJson,
        headword: str = "bank",
        forms: Iterable[RawJson] = (),
        minimum_year: int | None = None,
        maximum_year: int | None = None,
    ) -> list[Sentence]:
        sense: RawJson = {"glosses": ["A meaning."], "examples": list(examples)}
        entry: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "forms": list(forms),
            "senses": [sense],
        }

        lemmas = extract(
            [entry],
            minimum_year=minimum_year,
            maximum_year=maximum_year,
        )

        return lemmas[0].senses[0].sentences

    return run


class TestOpening:
    """
    Reading the file however it was compressed.
    """

    @given(st.lists(raw_entries(), max_size=3))
    def test_reads_the_same_entries_whatever_the_suffix_names(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """A parse writes zstd, but a file found elsewhere may be plain or gzipped."""
        plain = extract(entries, name="wiktextract.jsonl")

        assert extract(entries, name="wiktextract.jsonl.zst") == plain
        assert extract(entries, name="wiktextract.jsonl.gz") == plain


class TestEntries:
    """
    Which entries are read at all.
    """

    @given(st.data())
    def test_keeps_the_language_asked_for(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """A dump holds every language Wiktionary describes, not the one alone."""
        editions = data.draw(st.lists(languages, min_size=1, max_size=3, unique=True))
        entries = data.draw(
            st.lists(raw_entries(languages=st.sampled_from(editions)), max_size=5)
        )

        language = data.draw(st.sampled_from(editions))
        spoken = [entry for entry in entries if entry["lang_code"] == language]

        assert extract(entries, language=language) == extract(
            spoken,
            language=language,
        )

    @given(st.lists(raw_entries(), max_size=5), st.data())
    def test_keeps_only_the_parts_of_speech_asked_for(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """The filter narrows what is kept, and never widens it."""
        allowed = data.draw(st.sets(parts_of_speech, min_size=1))

        assert extract(entries, allowed_pos=frozenset(allowed)) == [
            lemma for lemma in extract(entries) if lemma.pos in allowed
        ]

    @given(st.lists(raw_entries(), max_size=4), _UNREADABLE, st.data())
    def test_reads_past_an_entry_it_cannot_collect(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
        unreadable: RawJson,
        data: st.DataObject,
    ) -> None:
        """An entry with no headword, no part of speech kept or no gloss is skipped."""
        position = data.draw(st.integers(min_value=0, max_value=len(entries)))
        mixed = [*entries[:position], unreadable, *entries[position:]]

        assert extract(mixed) == extract(entries)

    @given(st.lists(raw_entries(), max_size=4), _REPORTS, st.data())
    def test_reads_past_a_line_that_is_not_an_entry(
        self,
        extract_lines: Callable[[Iterable[str]], list[Lemma]],
        entries: list[RawJson],
        report: str,
        data: st.DataObject,
    ) -> None:
        """Wiktextract reports itself among the entries, so not every line is one."""
        position = data.draw(st.integers(min_value=0, max_value=len(entries)))
        lines = [json.dumps(entry) for entry in entries]

        assert extract_lines([*lines[:position], report, *lines[position:]]) == (
            extract_lines(lines)
        )

    @given(st.lists(raw_entries(headwords=_PADDED_HEADWORDS), max_size=4))
    def test_strips_the_headword_it_reads(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """The whitespace an editor left around a headword is not part of it."""
        assert [lemma.lemma for lemma in extract(entries)] == [
            str(entry["word"]).strip() for entry in entries
        ]

    @given(parts_of_speech, st.data())
    def test_reads_the_part_of_speech_off_the_code_wiktextract_writes(
        self,
        extract: Callable[..., list[Lemma]],
        pos: POS,
        data: st.DataObject,
    ) -> None:
        """The values are wiktextract's own codes, so a code converts directly."""
        entry = data.draw(raw_entries(pos_codes=st.just(pos.value)))

        assert extract([entry])[0].pos is pos


class TestIdentifiers:
    """
    How a lemma and its senses are named.
    """

    @given(st.lists(raw_entries(), max_size=5))
    def test_names_a_lemma_after_its_headword_and_part_of_speech(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """An identifier reads as bank.noun.2, the ordinal counting what is kept."""
        ordinals: Counter[str] = Counter()

        for lemma in extract(entries):
            key = f"{lemma.lemma}.{lemma.pos}"
            ordinals[key] += 1

            assert lemma.id == f"{key}.{ordinals[key]}"

    @given(st.lists(raw_entries(), max_size=5))
    def test_names_a_sense_after_its_lemma_and_its_position(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """A sense identifier reads as bank.noun.1.03, padded so that it sorts."""
        for lemma in extract(entries):
            assert [sense.id for sense in lemma.senses] == [
                f"{lemma.id}.{position:02d}"
                for position in range(1, len(lemma.senses) + 1)
            ]


class TestSenses:
    """
    What a sense carries over.
    """

    @given(st.lists(st.one_of(glosses, blanks), min_size=1, max_size=4), st.data())
    def test_keeps_the_gloss_chain_outermost_first(
        self,
        extract: Callable[..., list[Lemma]],
        chain: list[str],
        data: st.DataObject,
    ) -> None:
        """The chain is what lets a sub-sense be read on its own, whitespace aside."""
        senses: list[RawJson] = [{"glosses": chain}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        kept = tuple(gloss.strip() for gloss in chain if gloss.strip())

        assert [
            sense.glosses for lemma in extract([entry]) for sense in lemma.senses
        ] == ([kept] if kept else [])

    @given(
        st.lists(st.lists(glosses, min_size=1, max_size=3), min_size=1, max_size=4),
        st.data(),
    )
    def test_keeps_a_sub_sense_alongside_its_parent(
        self,
        extract: Callable[..., list[Lemma]],
        chains: list[list[str]],
        data: st.DataObject,
    ) -> None:
        """A parent is not replaced by what nests under it: it has examples too."""
        senses: list[RawJson] = [{"glosses": chain} for chain in chains]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        kept = extract([entry])[0].senses

        assert [sense.depth for sense in kept] == [len(chain) for chain in chains]
        assert [sense.gloss for sense in kept] == [
            chain[-1].strip() for chain in chains
        ]

    @given(st.lists(words, max_size=3), st.lists(words, max_size=3), st.data())
    def test_keeps_tags_and_topics_apart(
        self,
        extract: Callable[..., list[Lemma]],
        tags: list[str],
        topics: list[str],
        data: st.DataObject,
    ) -> None:
        """Labels of grammar are not subject fields, and neither takes the other."""
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(tags=st.just(tags), topics=st.just(topics)),
                    min_size=1,
                    max_size=1,
                )
            )
        )

        sense = extract([entry])[0].senses[0]

        assert (sense.tags, sense.topics) == (tuple(tags), tuple(topics))


class TestSentences:
    """
    The sentences illustrating a sense.
    """

    @given(st.lists(raw_examples(), max_size=4))
    def test_a_sentence_with_no_reference_is_an_example(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
    ) -> None:
        """An editor wrote it, so there is no source to name."""
        found = attest(*examples)

        assert all(isinstance(sentence, Example) for sentence in found)
        assert [sentence.text for sentence in found] == [
            str(example["text"]).strip() for example in examples
        ]

    @given(st.data())
    def test_a_sentence_with_a_reference_is_a_quotation(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """A reference is what makes a sentence evidence from somewhere."""
        year = data.draw(years)
        reference = data.draw(references(year))
        example = data.draw(raw_examples(references=st.just(reference)))

        quotation = attest(example)[0]

        assert isinstance(quotation, Quotation)
        assert (quotation.text, quotation.reference) == (
            str(example["text"]).strip(),
            reference,
        )

    @given(st.lists(st.one_of(texts, blanks), max_size=4), st.data())
    def test_strips_a_sentence_and_drops_a_blank_one(
        self,
        attest: Callable[..., list[Sentence]],
        written: list[str],
        data: st.DataObject,
    ) -> None:
        """A sentence of whitespace illustrates nothing."""
        examples = [data.draw(raw_examples(texts=st.just(text))) for text in written]

        assert [sentence.text for sentence in attest(*examples)] == [
            text.strip() for text in written if text.strip()
        ]


class TestYears:
    """
    Reading a year off a reference, and filtering on it.
    """

    @given(st.data())
    def test_reads_the_year_the_reference_names(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """A reference is prose, so the year is taken where it is recognised."""
        year = data.draw(years)
        example = data.draw(raw_examples(references=references(year)))

        quotation = attest(example)[0]

        assert isinstance(quotation, Quotation)
        assert quotation.year == year

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
    def test_reads_the_dates_wiktionary_writes(
        self,
        attest: Callable[..., list[Sentence]],
        reference: str,
        expected: int | None,
    ) -> None:
        """The shapes are Wiktionary's own, and the year opens each of them."""
        quotation = attest({"text": "He runs.", "ref": reference})[0]

        assert isinstance(quotation, Quotation)
        assert quotation.year == expected

    @given(st.lists(years, max_size=4), st.none() | years, st.none() | years, st.data())
    def test_keeps_the_quotations_inside_the_bounds(
        self,
        attest: Callable[..., list[Sentence]],
        dated: list[int],
        minimum_year: int | None,
        maximum_year: int | None,
        data: st.DataObject,
    ) -> None:
        """Both bounds are inclusive, and either stands on its own."""
        examples = [
            data.draw(raw_examples(references=references(year))) for year in dated
        ]

        kept = attest(
            *examples,
            minimum_year=minimum_year,
            maximum_year=maximum_year,
        )

        assert [
            sentence.year for sentence in kept if isinstance(sentence, Quotation)
        ] == [
            year
            for year in dated
            if (minimum_year is None or year >= minimum_year)
            and (maximum_year is None or year <= maximum_year)
        ]

    @given(st.lists(raw_examples(), max_size=4), st.none() | years, st.data())
    def test_a_bound_never_widens_what_is_kept(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
        minimum_year: int | None,
        data: st.DataObject,
    ) -> None:
        """A bound is asked for to leave something out, never to let something in."""
        dated = [
            data.draw(raw_examples(references=references(data.draw(years))))
            for _ in examples
        ]

        every = attest(*examples, *dated)
        bounded = attest(*examples, *dated, minimum_year=minimum_year)

        assert set(bounded) <= set(every)

    @given(st.data())
    def test_drops_an_undated_quotation_once_a_bound_is_set(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """A quotation nothing can date cannot be shown to be inside a bound."""
        undated = data.draw(raw_examples(references=undated_references))

        assert attest(undated, minimum_year=data.draw(years)) == []

    @given(st.data())
    def test_keeps_an_undated_quotation_when_no_bound_is_set(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """An undated quotation only stands in the way once a bound is set."""
        reference = data.draw(undated_references)
        undated = data.draw(raw_examples(references=st.just(reference)))

        quotation = attest(undated)[0]

        assert isinstance(quotation, Quotation)
        assert (quotation.reference, quotation.year) == (reference.strip(), None)

    @given(st.lists(raw_examples(), max_size=3), st.data())
    def test_leaves_examples_alone(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """The bounds reach quotations alone, an example carrying no date."""
        year = data.draw(years)
        outside = data.draw(raw_examples(references=references(year)))

        assert attest(*examples, outside, minimum_year=year + 1) == attest(*examples)

    @given(st.data())
    def test_a_sense_the_bounds_emptied_is_kept_all_the_same(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """A sense is what its gloss says, whatever evidence the bounds leave it."""
        year = data.draw(years)
        example = data.draw(raw_examples(references=references(year)))
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(examples=st.just([example])),
                    min_size=1,
                    max_size=1,
                )
            )
        )

        lemmas = extract([entry], minimum_year=year + 1)

        assert [sense.sentences for sense in lemmas[0].senses] == [[]]


class TestWordOffsets:
    """
    Where the lemma occurs in the sentences attesting it.
    """

    @given(words, st.data())
    def test_a_sentence_carries_where_the_lemma_occurs(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """The headword is a form of itself, so it is looked for like the rest."""
        example = data.draw(raw_examples(texts=st.just(f"1 {headword} 2")))

        assert attest(example, headword=headword)[0].word_offsets == (
            (2, 2 + len(headword)),
        )

    @given(words, words, st.data())
    def test_locates_an_inflection_wiktextract_listed(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        inflection: str,
        data: st.DataObject,
    ) -> None:
        """A sentence attests the lemma in whatever form it needs."""
        form = data.draw(raw_forms(forms=_padded(inflection)))
        example = data.draw(raw_examples(texts=st.just(f"1 {inflection} 2")))

        found = attest(example, headword=headword, forms=[form])

        assert (2, 2 + len(inflection)) in found[0].word_offsets

    @given(words, _SERVICE_TAGS, st.data())
    def test_skips_what_is_listed_among_the_forms_without_being_one(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        tag: str,
        data: st.DataObject,
    ) -> None:
        """An inflection table names itself, its template and its transliterations."""
        listed = data.draw(
            words.filter(lambda form: form.casefold() != headword.casefold())
        )

        form = data.draw(raw_forms(forms=st.just(listed), tags=st.just([tag])))
        example = data.draw(raw_examples(texts=st.just(f"1 {listed} 2")))

        found = attest(example, headword=headword, forms=[form])

        assert (2, 2 + len(listed)) not in found[0].word_offsets

    @given(words, _EMPTY_CELLS, st.data())
    def test_skips_a_form_the_inflection_table_left_empty(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        cell: str,
        data: st.DataObject,
    ) -> None:
        """A dash stands for a form that does not exist, and blank for none at all."""
        form = data.draw(raw_forms(forms=st.just(cell)))
        example = data.draw(raw_examples(texts=st.just(f"1 {headword} - 2")))

        found = attest(example, headword=headword, forms=[form])

        assert found[0].word_offsets == ((2, 2 + len(headword)),)
