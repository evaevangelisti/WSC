"""Tests for src/wsc/extract/wiktionary/."""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from kwic import Locator
from strategies import (
    RawJson,
    blanks,
    form_tags,
    glosses,
    languages,
    parts_of_speech,
    raw_entries,
    raw_examples,
    raw_forms,
    raw_senses,
    raw_synonyms,
    raw_translations,
    references,
    sentence_kinds,
    texts,
    undated_references,
    unknown_pos_codes,
    words,
    years,
)

from wsc.constants import LANGUAGE
from wsc.extract import WiktionaryExtractor
from wsc.models import (
    POS,
    Example,
    Lemma,
    Quotation,
    Sentence,
    WordOffset,
    WordOffsetSource,
)

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


_UNREADABLE = st.one_of(
    raw_entries(headwords=st.just("") | blanks),
    raw_entries(pos_codes=unknown_pos_codes),
    raw_entries(
        senses=st.lists(raw_senses(glosses=st.lists(blanks, max_size=2)), max_size=2)
    ),
)

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

_SERVICE_TAGS = st.sampled_from(["inflection-template", "romanization", "table-tags"])

_EMPTY_CELLS = st.sampled_from(["-", ""]) | blanks

_PADDING = blanks | st.just("")


def _pointer(
    headword: str,
) -> str:
    """
    Write what Wiktionary's seeCites template leaves in place of a sentence.

    Args:
        headword: The word whose citations page is pointed at.

    Returns:
        The line, as wiktextract passes it on.
    """
    return f"For quotations using this term, see Citations:{headword}."


@pytest.fixture
def extract(
    workspace: Callable[[], Path],
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
    locator: Locator,
) -> Callable[..., list[Lemma]]:
    """
    Run an extraction over entries a property drew.

    Args:
        workspace: Sets aside a directory for the file being read.
        write_entries: Writes the entries where the extractor will read them.
        locator: The search the extraction locates the lemma with.

    Returns:
        A runner extracting lemmas under the supplied filters.
    """

    def run(
        entries: Iterable[RawJson],
        allowed_pos: frozenset[POS] | None = None,
        minimum_year: int | None = None,
        maximum_year: int | None = None,
        name: str = "wiktextract.jsonl",
    ) -> list[Lemma]:
        path = write_entries(workspace() / name, entries)

        extractor = WiktionaryExtractor(
            allowed_pos,
            minimum_year,
            maximum_year,
            locator,
        )

        return list(extractor.extract(path))

    return run


@pytest.fixture
def extract_lines(
    workspace: Callable[[], Path],
    locator: Locator,
) -> Callable[[Iterable[str]], list[Lemma]]:
    """
    Run an extraction over the lines of a file, entries or otherwise.

    Args:
        workspace: Sets aside a directory for the file being read.
        locator: The search the extraction locates the lemma with.

    Returns:
        A runner extracting entries from supplied JSONL lines.
    """

    def run(
        lines: Iterable[str],
    ) -> list[Lemma]:
        path = workspace() / "wiktextract.jsonl"
        _ = path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")

        return list(WiktionaryExtractor(None, None, None, locator).extract(path))

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
        A runner filtering examples by headword, forms, and quotation years.
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
    """Reading the file however it was compressed."""

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
    """Which entries are read at all."""

    @given(st.data())
    def test_keeps_english_alone(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """The extractor keeps English entries from multilingual dumps."""
        editions = data.draw(
            st.lists(languages, min_size=1, max_size=3, unique=True).map(
                lambda drawn: [*drawn, LANGUAGE]
            )
        )
        entries = data.draw(
            st.lists(raw_entries(languages=st.sampled_from(editions)), max_size=5)
        )

        spoken = [entry for entry in entries if entry["lang_code"] == LANGUAGE]

        assert extract(entries) == extract(spoken)

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
        gathered = dict.fromkeys(
            (str(entry["word"]).strip(), entry["pos"]) for entry in entries
        )

        assert [lemma.lemma for lemma in extract(entries)] == [
            headword for headword, _ in gathered
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
    """How a lemma and its senses are named."""

    @given(st.lists(raw_entries(), max_size=5))
    def test_opens_an_identifier_with_the_headword_and_part_of_speech(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """An entry is named bank.noun, and a sense adds a digest of its own."""
        for lemma in extract(entries):
            key = f"{lemma.lemma}.{lemma.pos}"

            assert lemma.id == key
            assert all(sense.id.startswith(f"{key}.") for sense in lemma.senses)

    @given(st.lists(raw_entries(), max_size=5))
    def test_names_a_sense_after_what_it_says(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """Two senses reading the same way are the same sense, wherever they sit."""
        named: dict[str, tuple[str, ...]] = {}

        for lemma in extract(entries):
            for sense in lemma.senses:
                assert named.setdefault(sense.id, sense.glosses) == sense.glosses

    @given(st.lists(raw_senses(), min_size=1, max_size=4), st.data())
    def test_names_a_sense_the_same_however_the_entry_is_ordered(
        self,
        extract: Callable[..., list[Lemma]],
        senses: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """A page reordered upstream reads back under the identifiers it had."""
        headword = data.draw(words)
        entry: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": senses,
        }
        reordered: RawJson = {**entry, "senses": list(reversed(senses))}

        read = {sense.id for lemma in extract([entry]) for sense in lemma.senses}
        again = {sense.id for lemma in extract([reordered]) for sense in lemma.senses}

        assert read == again


class TestSenses:
    """What a sense carries over."""

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

        gathered = list(
            dict.fromkeys(
                tuple(gloss.strip() for gloss in chain if gloss.strip())
                for chain in chains
            )
        )

        kept = extract([entry])[0].senses

        assert [sense.depth for sense in kept] == [len(chain) for chain in gathered]
        assert [sense.gloss for sense in kept] == [chain[-1] for chain in gathered]

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


class TestPseudoSenses:
    """Exclude senses describing inflected forms."""

    @given(form_tags, st.data())
    def test_drops_a_sense_that_only_inflects_the_headword(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        data: st.DataObject,
    ) -> None:
        """Inflection senses point to definitions under another entry."""
        senses: list[RawJson] = [{"glosses": ["Plural of bank."], "tags": [tag]}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        assert extract([entry]) == []

    @given(form_tags, st.lists(st.booleans(), min_size=1, max_size=6), st.data())
    def test_names_what_is_kept_without_naming_what_is_not(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        inflecting: list[bool],
        data: st.DataObject,
    ) -> None:
        """A form is no sense, so it leaves no identifier behind."""
        senses: list[RawJson] = [
            {"glosses": [f"Sense {position}."], **({"tags": [tag]} if drop else {})}
            for position, drop in enumerate(inflecting)
        ]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        lemmas = extract([entry])
        kept = lemmas[0].senses if lemmas else []

        assert [sense.gloss for sense in kept] == [
            f"Sense {position}." for position, drop in enumerate(inflecting) if not drop
        ]
        assert all(
            sense.id.startswith(f"{lemmas[0].lemma}.{lemmas[0].pos}.") for sense in kept
        )

    @given(form_tags, st.data())
    def test_an_entry_of_forms_alone_is_left_out(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        data: st.DataObject,
    ) -> None:
        """Only the entry that defines something comes through."""
        headword = data.draw(words)
        inflected: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["Plural of bank."], "tags": [tag]}],
        }
        defined: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["A meaning."]}],
        }

        lemmas = extract([inflected, defined])

        assert [lemma.lemma for lemma in lemmas] == [headword]

    @given(st.lists(words, max_size=3), st.data())
    def test_keeps_a_sense_whose_tags_say_nothing_about_forms(
        self,
        extract: Callable[..., list[Lemma]],
        tags: list[str],
        data: st.DataObject,
    ) -> None:
        """Only the two inflection tags exclude a sense."""
        senses: list[RawJson] = [{"glosses": ["A meaning."], "tags": tags}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        assert [sense.tags for sense in extract([entry])[0].senses] == [tuple(tags)]


class TestSentences:
    """The sentences illustrating a sense."""

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


class TestKinds:
    """Reading a sentence as the kind wiktextract says it is."""

    @given(st.data())
    def test_wiktextract_settles_the_kind_over_a_reference_it_withheld(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """It read the markup; a reference is only what it left behind."""
        year = data.draw(years)
        reference = data.draw(references(year))
        text = data.draw(texts)

        quoted = attest({"text": f"{reference}\n{text}", "type": "quotation"})[0]
        plain = attest({"text": f"{reference}\n{text}", "type": "example"})[0]

        assert isinstance(quoted, Quotation)
        assert isinstance(plain, Example)

    @given(texts)
    def test_a_quotation_with_no_source_to_tell_apart_is_kept_as_an_example(
        self,
        attest: Callable[..., list[Sentence]],
        written: str,
    ) -> None:
        """An export tells the two apart by the reference, and it has none."""
        found = attest({"text": written, "type": "quotation"})[0]

        assert isinstance(found, Example)
        assert found.text == written.strip()

    @given(st.data())
    def test_a_quotation_naming_no_source_is_still_a_quotation(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """The source is in the text, which is why no reference came with it."""
        year = data.draw(years)
        reference = data.draw(references(year))
        text = data.draw(texts)

        found = attest(
            {"text": f"{reference}\n{text}", "type": "quotation"},
        )[0]

        assert isinstance(found, Quotation)
        assert (found.text, found.reference, found.year) == (
            text.strip(),
            reference,
            year,
        )

    @given(st.data())
    def test_an_undated_first_line_is_a_source_only_where_it_says_so(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """A break alone proves nothing: prose runs over lines too."""
        head = data.draw(undated_references)
        tail = data.draw(texts)
        written = f"{head}\n{tail}"

        untyped = attest({"text": written})[0]

        assert isinstance(untyped, Example)
        assert untyped.text == written.strip()

        quoted = attest({"text": written, "type": "quotation"})[0]

        assert isinstance(quoted, Quotation)
        assert (quoted.text, quoted.reference) == (tail.strip(), head.strip())

    @given(st.data())
    def test_a_sentence_read_as_an_example_is_never_split(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """Its own word comes first, whatever its opening line looks like."""
        year = data.draw(years)
        written = f"{data.draw(references(year))}\n{data.draw(texts)}"

        found = attest({"text": written, "type": "example"})[0]

        assert isinstance(found, Example)
        assert found.text == written.strip()

    @given(st.data())
    def test_a_break_with_nothing_after_it_leaves_no_sentence(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """Splitting there would leave the reference, which attests nothing."""
        year = data.draw(years)
        written = f"{year}, A Book\n{data.draw(blanks)}"

        assert attest({"text": written, "type": "quotation"}) == []

    @given(st.data())
    def test_a_source_split_off_is_left_out_of_the_offsets(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """A headword named in a book title is not an occurrence of it."""
        headword = data.draw(words)
        year = data.draw(years)
        reference = f"{year}, {headword}, A Book"

        found = attest(
            {"text": f"{reference}\n{headword}", "type": "quotation"},
            headword=headword,
        )[0]

        assert found.text == headword
        assert found.word_offsets == (
            WordOffset(
                (0, len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )


class TestPointers:
    """Passing over what stands in for a sentence without being one."""

    @given(words)
    def test_drops_a_pointer_to_the_citations_page(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
    ) -> None:
        """It navigates somewhere; it attests nothing."""
        assert attest({"text": _pointer(headword)}, headword=headword) == []

    @given(words, sentence_kinds)
    def test_keeps_a_sentence_wiktextract_read_as_one(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        kind: str,
    ) -> None:
        """A pointer is left where no kind was read, so a kind rules it out."""
        written = _pointer(headword)

        found = attest({"text": written, "type": kind}, headword=headword)

        assert [sentence.text for sentence in found] == [written]

    @given(words, st.data())
    def test_keeps_a_sentence_that_merely_opens_the_same_way(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """Pointer detection requires the complete template."""
        written = f"For quotations {data.draw(texts)}"

        found = attest({"text": written}, headword=headword)

        assert [sentence.text for sentence in found] == [written.strip()]


class TestYears:
    """Reading a year off a reference, and filtering on it."""

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
    """Where the lemma occurs in the sentences attesting it."""

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
            WordOffset(
                (2, 2 + len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )

    def test_combines_sources_supporting_the_same_offset(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """Agreement remains distinguishable from either method alone."""
        found = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[2, 6]],
            },
        )

        assert found[0].word_offsets == (
            WordOffset(
                (2, 6),
                (
                    WordOffsetSource.BOLD,
                    WordOffsetSource.LEMMATIZER,
                ),
            ),
        )

    def test_keeps_disagreeing_offsets_separately(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """A later review needs both proposals where the methods disagree."""
        found = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[7, 14]],
            },
        )

        assert found[0].word_offsets == (
            WordOffset((2, 6), (WordOffsetSource.LEMMATIZER,)),
            WordOffset((7, 14), (WordOffsetSource.BOLD,)),
        )

    def test_discards_a_bold_offset_outside_the_sentence(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """An invalid source range cannot identify text for review."""
        found = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[2, 100]],
            },
        )

        assert found[0].word_offsets == (
            WordOffset((2, 6), (WordOffsetSource.LEMMATIZER,)),
        )

    def test_moves_bold_offsets_with_a_removed_reference(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """Bold ranges remain relative to the exported sentence."""
        reference = "2026, bank"
        start = len(reference) + 1
        found = attest(
            {
                "text": f"{reference}\nbank",
                "type": "quotation",
                "bold_text_offsets": [[start, start + 4]],
            },
        )

        assert found[0].word_offsets == (
            WordOffset(
                (0, 4),
                (
                    WordOffsetSource.BOLD,
                    WordOffsetSource.LEMMATIZER,
                ),
            ),
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

        assert (2, 2 + len(inflection)) in (
            word_offset.offset for word_offset in found[0].word_offsets
        )

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

        assert (2, 2 + len(listed)) not in (
            word_offset.offset for word_offset in found[0].word_offsets
        )

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

        assert found[0].word_offsets == (
            WordOffset(
                (2, 2 + len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )


class TestVariants:
    """How else a lemma is spelled, which one thing says."""

    @given(
        words,
        st.lists(words, min_size=1, max_size=3).map(".".join),
        parts_of_speech,
        st.data(),
    )
    def test_takes_a_spelling_from_the_entry_pointing_at_it(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        pos: POS,
        data: st.DataObject,
    ) -> None:
        """Another spelling sits on a page of its own and points back."""
        pointing: RawJson = {
            "word": spelling,
            "pos": pos.value,
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Alternative spelling of {headword}."],
                    "tags": ["alt-of"],
                    "alt_of": [{"word": headword}],
                }
            ],
        }
        defined = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just(pos.value))
        )

        lemmas = extract([pointing, defined])

        assert [lemma.variants for lemma in lemmas] == [
            frozenset({f"{spelling}.{pos}"}) - {f"{headword}.{pos}"}
        ]

    @given(words, words, st.data())
    def test_leaves_an_inflection_alone(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """Plural forms are excluded from spelling variants."""
        pointing: RawJson = {
            "word": spelling,
            "pos": "noun",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Plural of {headword}."],
                    "tags": ["form-of", "plural"],
                    "form_of": [{"word": headword}],
                }
            ],
        }
        defined = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just("noun"))
        )

        lemmas = extract([pointing, defined])

        assert [lemma.variants for lemma in lemmas] == [frozenset()]

    @given(words, words, st.data())
    def test_leaves_the_forms_an_entry_lists_for_itself_alone(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """An Alternative forms section names derivations as readily as spellings."""
        form = data.draw(
            raw_forms(forms=st.just(spelling), tags=st.just(["alternative"]))
        )
        entry = data.draw(
            raw_entries(headwords=st.just(headword), forms=st.just([form]))
        )

        (lemma,) = extract([entry])

        assert lemma.variants == frozenset()

    @given(words, words, st.data())
    def test_points_only_at_the_same_part_of_speech(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """A spelling of the noun says nothing about how the verb is written."""
        pointing: RawJson = {
            "word": spelling,
            "pos": "verb",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Alternative spelling of {headword}."],
                    "tags": ["alt-of"],
                    "alt_of": [{"word": headword}],
                }
            ],
        }
        defined = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just("noun"))
        )

        lemmas = extract([pointing, defined])

        assert [lemma.variants for lemma in lemmas] == [frozenset()]


class TestTranslations:
    """What other languages call the entry, which Wiktionary hangs off the entry."""

    @given(words, languages, glosses, st.data())
    def test_gathers_a_translation_under_the_gloss_and_the_language(
        self,
        extract: Callable[..., list[Lemma]],
        translation: str,
        language: str,
        gloss: str,
        data: st.DataObject,
    ) -> None:
        """The gloss is stripped, an editor having written it by hand."""
        raw = data.draw(
            raw_translations(
                translations=st.just(translation),
                codes=st.just(language),
                glosses=st.just(gloss),
            )
        )
        entry = data.draw(raw_entries(translations=st.just([raw])))

        (lemma,) = extract([entry])

        assert lemma.translation_tables[0].gloss == gloss.strip()
        assert lemma.translation_tables[0].translations == {
            language: frozenset({translation})
        }

    @given(words, words, languages, glosses, st.data())
    def test_gathers_two_words_of_one_language_together(
        self,
        extract: Callable[..., list[Lemma]],
        first: str,
        second: str,
        language: str,
        gloss: str,
        data: st.DataObject,
    ) -> None:
        """One meaning is often said more than one way in the same language."""
        drawn = [
            data.draw(
                raw_translations(
                    translations=st.just(word),
                    codes=st.just(language),
                    glosses=st.just(gloss),
                )
            )
            for word in (first, second)
        ]
        entry = data.draw(raw_entries(translations=st.just(drawn)))

        (lemma,) = extract([entry])

        table = next(
            table for table in lemma.translation_tables if table.gloss == gloss.strip()
        )
        assert table.translations[language] == frozenset({first, second})

    @given(st.sampled_from(("word", "lang_code", "sense")), st.data())
    def test_leaves_out_a_translation_missing_what_keys_it(
        self,
        extract: Callable[..., list[Lemma]],
        key: str,
        data: st.DataObject,
    ) -> None:
        """A translation is filed under its gloss and its language, or nowhere."""
        raw = data.draw(raw_translations())
        raw[key] = data.draw(blanks)

        entry = data.draw(raw_entries(translations=st.just([raw])))

        (lemma,) = extract([entry])

        assert not lemma.translation_tables

    @given(st.data())
    def test_carries_none_where_the_entry_lists_none(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """Wiktionary writes a translation table for a fraction of its entries."""
        entry = data.draw(raw_entries(translations=st.just([])))

        (lemma,) = extract([entry])

        assert not lemma.translation_tables


class TestSynonyms:
    """Other words standing for what a sense means."""

    @given(words, words, st.data())
    def test_ties_a_synonym_to_the_sense_that_names_it(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        synonym: str,
        data: st.DataObject,
    ) -> None:
        """A synonym under a sense stands for that meaning alone."""
        raw = data.draw(raw_synonyms(words=st.just(synonym)))
        sense = data.draw(raw_senses(synonyms=st.just([raw])))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense]))
        )

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == (() if synonym == headword else (synonym,))

    @given(words, words, words, st.data())
    def test_keeps_two_synonyms_in_the_order_they_were_listed(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        first: str,
        second: str,
        data: st.DataObject,
    ) -> None:
        """A sense may be put more than one way, and order is what is read."""
        raw = [data.draw(raw_synonyms(words=st.just(word))) for word in (first, second)]
        sense = data.draw(raw_senses(synonyms=st.just(raw)))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense]))
        )

        (lemma,) = extract([entry])

        # Generated words may coincide; deduplication preserves their first occurrence.
        expected: dict[str, None] = {}

        for word in (first, second):
            if word != headword:
                expected[word] = None

        assert lemma.senses[0].synonyms == tuple(expected)

    @given(words, st.data())
    def test_leaves_out_the_headword_as_a_synonym_of_its_own_sense(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """A word is not offered as another way to say itself."""
        raw = data.draw(raw_synonyms(words=st.just(headword)))
        sense = data.draw(raw_senses(synonyms=st.just([raw])))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense]))
        )

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ()

    @given(st.data())
    def test_carries_none_where_the_sense_lists_none(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """Most senses offer no synonym at all."""
        sense = data.draw(raw_senses(synonyms=st.just([])))
        entry = data.draw(raw_entries(senses=st.just([sense])))

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ()


class TestStrayReferences:
    """Passing over a reference left standing where a sentence belongs."""

    @given(years)
    def test_leaves_out_a_reference_nobody_split_off(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
    ) -> None:
        """A citation on one line names a source and attests nothing."""
        assert attest({"text": f"{year}, John Milton, A Book"}) == []

    @given(years)
    def test_keeps_a_sentence_opening_on_a_date(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
    ) -> None:
        """A year is how a sentence about a year opens, and it is a sentence."""
        written = f"{year} saw the release of many great films."

        assert attest({"text": written})[0].text == written

    @given(years, st.data())
    def test_keeps_a_reference_a_body_follows(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
        data: st.DataObject,
    ) -> None:
        """A line break separates the reference from its sentence."""
        written = f"{year}, A Book\n{data.draw(texts)}"

        assert attest({"text": written, "type": "quotation"})[0].text
