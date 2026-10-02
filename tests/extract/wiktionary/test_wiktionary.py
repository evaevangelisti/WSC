"""
Tests for src/wsc/extract/wiktionary/.
"""

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from kwic import Locator
from strategies import (
    RawJson,
    blanks,
    definitions,
    form_tags,
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
from wsc.extract.markup import normalize_formatting, normalize_statement
from wsc.extract.wiktionary.parts.sentences import clean_reference, clean_sentence
from wsc.models import (
    POS,
    Attestation,
    Example,
    Language,
    Lemma,
    Quotation,
    Sentence,
    WordOffset,
    WordOffsetSource,
)

_loads: Callable[[str], object] = json.loads


def _has_lexical_content(
    text: str,
) -> bool:
    """
    Return whether text contains at least one letter or number.

    Args:
        text: Candidate sentence or synonym.

    Returns:
        Whether the text is more than punctuation.
    """
    return any(character.isalnum() for character in text)


_LEXICAL_TEXTS = texts.filter(_has_lexical_content)

_TRANSLATION_GLOSSES = st.lists(words, min_size=1, max_size=4).map(
    lambda terms: f"A meaning {' '.join(terms)}",
)


def _is_json_entry(
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


def _pad_word(
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


_PADDED_HEADWORDS = words.flatmap(_pad_word)


_UNREADABLE = st.one_of(
    raw_entries(headwords=st.just("") | blanks),
    raw_entries(pos_codes=unknown_pos_codes),
    raw_entries(
        senses=st.lists(raw_senses(glosses=st.lists(blanks, max_size=2)), max_size=2),
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
        ],
    ),
).filter(
    lambda line: not _is_json_entry(line),
)

_SERVICE_TAGS = st.sampled_from(["inflection-template", "romanization", "table-tags"])

_EMPTY_CELLS = st.sampled_from(["-", ""]) | blanks

_PADDING = blanks | st.just("")


def _build_pointer(
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
        """
        Extract lemmas from generated Wiktextract entries.

        Args:
            entries: Source entries to serialize and extract.
            allowed_pos: Parts of speech to retain, or None for all supported ones.
            minimum_year: Oldest retained quotation year, or None.
            maximum_year: Newest retained quotation year, or None.
            name: Source filename selecting the compression format.

        Returns:
            Collected lemmas after filtering.
        """
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
        """
        Extract lemmas from supplied source lines.

        Args:
            lines: Source lines emitted or written in their supplied order.

        Returns:
            Collected lemmas from readable entries.
        """
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
        """
        Collect attestations for a single generated sense.

        Args:
            examples: Raw examples attached to the generated sense.
            headword: Lemma whose occurrences the extractor locates.
            forms: Inflected forms supplied with the entry.
            minimum_year: Oldest retained quotation year, or None.
            maximum_year: Newest retained quotation year, or None.

        Returns:
            Sentences retained by the configured filters.
        """
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
    def test_reads_compressed_entries(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """
        A parse writes zstd, but a file found elsewhere may be plain or gzipped.
        """
        plain_lemmas = extract(entries, name="wiktextract.jsonl")

        assert extract(entries, name="wiktextract.jsonl.zst") == plain_lemmas
        assert extract(entries, name="wiktextract.jsonl.gz") == plain_lemmas


class TestEntries:
    """
    Which entries are read at all.
    """

    @given(st.data())
    def test_selects_english_entries(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """
        The extractor keeps English entries from multilingual dumps.
        """
        editions = data.draw(
            st.lists(languages, min_size=1, max_size=3, unique=True).map(
                lambda drawn: [*drawn, LANGUAGE],
            ),
        )
        entries = data.draw(
            st.lists(raw_entries(languages=st.sampled_from(editions)), max_size=5),
        )

        english_entries = [entry for entry in entries if entry["lang_code"] == LANGUAGE]

        assert extract(entries) == extract(english_entries)

    @given(st.lists(raw_entries(), max_size=5), st.data())
    def test_filters_selected_categories(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """
        The filter narrows what is kept, and never widens it.
        """
        allowed = data.draw(st.sets(parts_of_speech, min_size=1))

        assert extract(entries, allowed_pos=frozenset(allowed)) == [
            lemma for lemma in extract(entries) if lemma.pos in allowed
        ]

    @given(st.lists(raw_entries(), max_size=4), _UNREADABLE, st.data())
    def test_skips_uncollectable_entries(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
        unreadable: RawJson,
        data: st.DataObject,
    ) -> None:
        """
        An entry with no headword, no part of speech kept or no gloss is skipped.
        """
        position = data.draw(st.integers(min_value=0, max_value=len(entries)))
        mixed = [*entries[:position], unreadable, *entries[position:]]

        assert extract(mixed) == extract(entries)

    @given(st.lists(raw_entries(), max_size=4), _REPORTS, st.data())
    def test_skips_invalid_lines(
        self,
        extract_lines: Callable[[Iterable[str]], list[Lemma]],
        entries: list[RawJson],
        report: str,
        data: st.DataObject,
    ) -> None:
        """
        Wiktextract reports itself among the entries, so not every line is one.
        """
        position = data.draw(st.integers(min_value=0, max_value=len(entries)))
        lines = [json.dumps(entry) for entry in entries]

        assert extract_lines([*lines[:position], report, *lines[position:]]) == (
            extract_lines(lines)
        )

    @given(st.lists(raw_entries(headwords=_PADDED_HEADWORDS), max_size=4))
    def test_strips_headword(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """
        The whitespace an editor left around a headword is not part of it.
        """
        entry_keys = dict.fromkeys(
            (str(entry["word"]).strip(), entry["pos"]) for entry in entries
        )

        assert [lemma.lemma for lemma in extract(entries)] == [
            headword for headword, _ in entry_keys
        ]

    @given(parts_of_speech, st.data())
    def test_maps_wiktextract_categories(
        self,
        extract: Callable[..., list[Lemma]],
        pos: POS,
        data: st.DataObject,
    ) -> None:
        """
        The values are wiktextract's own codes, so a code converts directly.
        """
        entry = data.draw(raw_entries(pos_codes=st.just(pos.value)))

        assert extract([entry])[0].pos is pos

    @given(raw_entries(pos_codes=st.just("name")))
    def test_maps_wiktextract_names(
        self,
        extract: Callable[..., list[Lemma]],
        entry: RawJson,
    ) -> None:
        """
        Wiktextract's name code represents a proper noun.
        """
        assert extract([entry])[0].pos is POS.PROPN


class TestIdentifiers:
    """
    How a lemma and its senses are named.
    """

    @given(st.lists(raw_entries(), max_size=5))
    def test_prefixes_entry_identifiers(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """
        An entry is named bank.noun, and a sense adds a digest of its own.
        """
        for lemma in extract(entries):
            entry_id = f"{lemma.lemma}.{lemma.pos}"

            assert lemma.id == entry_id
            assert all(sense.id.startswith(f"{entry_id}.") for sense in lemma.senses)

    @given(st.lists(raw_entries(), max_size=5))
    def test_hashes_sense_glosses(
        self,
        extract: Callable[..., list[Lemma]],
        entries: list[RawJson],
    ) -> None:
        """
        Two senses reading the same way are the same sense, wherever they sit.
        """
        sense_glosses: dict[str, tuple[str, ...]] = {}

        for lemma in extract(entries):
            for sense in lemma.senses:
                assert (
                    sense_glosses.setdefault(sense.id, sense.glosses) == sense.glosses
                )

    @given(st.lists(raw_senses(), min_size=1, max_size=4), st.data())
    def test_stabilizes_sense_identifiers(
        self,
        extract: Callable[..., list[Lemma]],
        senses: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """
        A page reordered upstream reads back under the identifiers it had.
        """
        headword = data.draw(words)
        entry: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": senses,
        }
        reordered: RawJson = {**entry, "senses": list(reversed(senses))}

        original_identifiers = {
            sense.id for lemma in extract([entry]) for sense in lemma.senses
        }
        reordered_identifiers = {
            sense.id for lemma in extract([reordered]) for sense in lemma.senses
        }

        assert original_identifiers == reordered_identifiers


class TestSenses:
    """
    What a sense carries over.
    """

    @given(st.lists(st.one_of(definitions, blanks), min_size=1, max_size=4), st.data())
    def test_preserves_gloss_hierarchy(
        self,
        extract: Callable[..., list[Lemma]],
        chain: list[str],
        data: st.DataObject,
    ) -> None:
        """
        Each gloss in a hierarchy retains its standalone cleaned meaning.
        """
        senses: list[RawJson] = [{"glosses": chain}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        expected_glosses = tuple(
            sense.gloss
            for gloss in chain
            if gloss.strip()
            for lemma in extract(
                [{**entry, "senses": [{"glosses": [gloss]}]}],
            )
            for sense in lemma.senses
        )

        assert [
            sense.glosses for lemma in extract([entry]) for sense in lemma.senses
        ] == ([expected_glosses] if expected_glosses else [])

    @given(
        st.lists(st.lists(definitions, min_size=1, max_size=3), min_size=1, max_size=4),
        st.data(),
    )
    def test_preserves_nested_senses(
        self,
        extract: Callable[..., list[Lemma]],
        chains: list[list[str]],
        data: st.DataObject,
    ) -> None:
        """
        A parent is not replaced by what nests under it: it has examples too.
        """
        senses: list[RawJson] = [{"glosses": chain} for chain in chains]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        normalized_chains = (
            tuple(normalize_statement(gloss) for gloss in chain if gloss.strip())
            for chain in chains
        )
        unique_chains = list(
            dict.fromkeys(
                normalized_chains,
            ),
        )

        collected_senses = extract([entry])[0].senses

        assert [sense.depth for sense in collected_senses] == [
            len(chain) for chain in unique_chains
        ]
        assert [sense.gloss for sense in collected_senses] == [
            chain[-1] for chain in unique_chains
        ]

    @given(st.lists(words, max_size=3), st.lists(words, max_size=3), st.data())
    def test_separates_sense_labels(
        self,
        extract: Callable[..., list[Lemma]],
        tags: list[str],
        topics: list[str],
        data: st.DataObject,
    ) -> None:
        """
        Labels of grammar are not subject fields, and neither takes the other.
        """
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(tags=st.just(tags), topics=st.just(topics)),
                    min_size=1,
                    max_size=1,
                ),
            ),
        )

        sense = extract([entry])[0].senses[0]

        assert (sense.tags, sense.topics) == (tuple(tags), tuple(topics))


class TestPseudoSenses:
    """
    Exclude senses describing inflected forms.
    """

    @given(
        descriptions=st.lists(
            st.tuples(
                st.sampled_from(
                    [
                        "Plural of bank",
                        "Dative of bank",
                        "Masculine plural",
                        "Misspelling of bank",
                        "Synonym of bank",
                        "Alternative spelling of bank",
                        "Obsolete spelling of bank",
                    ]
                ),
                st.booleans(),
            ),
            max_size=12,
        ),
        separator=st.sampled_from([" ", "  ", "\t", "\n"]),
    )
    def test_classifies_untagged_glosses(
        self,
        extract: Callable[..., list[Lemma]],
        descriptions: list[tuple[str, bool]],
        separator: str,
    ) -> None:
        """
        Untagged redirects are excluded only when they begin a hierarchy gloss.
        """
        senses: list[RawJson] = [
            {
                "glosses": [
                    f"Meaning {index}",
                    separator.join(description.upper().split())
                    if redirect
                    else f"A definition mentioning {description}",
                ],
            }
            for index, (description, redirect) in enumerate(descriptions)
        ]
        entry: RawJson = {
            "word": "bank",
            "pos": "noun",
            "lang_code": "en",
            "senses": senses,
        }
        collected = extract([entry])

        assert [sense.glosses[0] for lemma in collected for sense in lemma.senses] == [
            normalize_statement(f"Meaning {index}")
            for index, (_, redirect) in enumerate(descriptions)
            if not redirect
        ]

    @given(form_tags, st.data())
    def test_excludes_inflected_senses(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        data: st.DataObject,
    ) -> None:
        """
        Inflection senses point to definitions under another entry.
        """
        senses: list[RawJson] = [{"glosses": ["Plural of bank."], "tags": [tag]}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        assert extract([entry]) == []

    @given(form_tags, st.lists(st.booleans(), min_size=1, max_size=6), st.data())
    def test_preserves_retained_identifiers(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        inflecting: list[bool],
        data: st.DataObject,
    ) -> None:
        """
        A form is no sense, so it leaves no identifier behind.
        """
        senses: list[RawJson] = [
            {"glosses": [f"Sense {position}."], **({"tags": [tag]} if drop else {})}
            for position, drop in enumerate(inflecting)
        ]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        lemmas = extract([entry])
        collected_senses = lemmas[0].senses if lemmas else []

        assert [sense.gloss for sense in collected_senses] == [
            f"Sense {position}." for position, drop in enumerate(inflecting) if not drop
        ]
        assert all(
            sense.id.startswith(f"{lemmas[0].lemma}.{lemmas[0].pos}.")
            for sense in collected_senses
        )

    @given(form_tags, st.data())
    def test_excludes_inflected_entries(
        self,
        extract: Callable[..., list[Lemma]],
        tag: str,
        data: st.DataObject,
    ) -> None:
        """
        Only the entry that defines something comes through.
        """
        headword = data.draw(words)
        inflected_entry: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["Plural of bank."], "tags": [tag]}],
        }
        defining_entry: RawJson = {
            "word": headword,
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["A meaning."]}],
        }

        lemmas = extract([inflected_entry, defining_entry])

        assert [lemma.lemma for lemma in lemmas] == [headword]

    @given(st.lists(words, max_size=3), st.data())
    def test_preserves_defining_senses(
        self,
        extract: Callable[..., list[Lemma]],
        tags: list[str],
        data: st.DataObject,
    ) -> None:
        """
        Only the two inflection tags exclude a sense.
        """
        senses: list[RawJson] = [{"glosses": ["A meaning."], "tags": tags}]
        entry = data.draw(raw_entries(senses=st.just(senses)))

        assert [sense.tags for sense in extract([entry])[0].senses] == [tuple(tags)]


class TestSentences:
    """
    The sentences illustrating a sense.
    """

    def test_splits_layout_examples(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Supported em-space boundaries produce separate examples.
        """
        sentences = attest(
            {
                "text": "bank one\u2003 bank two",
                "bold_text_offsets": [[0, 4], [10, 14]],
            },
            {"text": "first bank;\u2003 second bank"},
        )

        assert [sentence.text for sentence in sentences] == [
            "bank one",
            "bank two",
            "first bank",
            "second bank",
        ]
        assert all(
            sentence.word_offsets[0].offset[0] < len(sentence.text)
            for sentence in sentences
        )

    def test_splits_semicolon_examples_with_independent_bold_ranges(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Separate examples flattened by Wiktextract when both name the lemma.
        """
        text = "she helped the blind man across;  the river is half a mile across"
        sentences = attest(
            {
                "text": text,
                "bold_text_offsets": [[25, 31], [59, 65]],
                "type": "example",
            },
            headword="across",
        )

        assert [sentence.text for sentence in sentences] == [
            "she helped the blind man across",
            "the river is half a mile across",
        ]
        assert [sentence.word_offsets[0].offset for sentence in sentences] == [
            (25, 31),
            (25, 31),
        ]

    def test_splits_double_spaced_semicolon_without_bold_ranges(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A double space supplies a boundary when Wiktextract kept the layout.
        """
        sentences = attest(
            {"text": "first bank;  second bank", "type": "example"},
        )

        assert [sentence.text for sentence in sentences] == [
            "first bank",
            "second bank",
        ]

    @pytest.mark.parametrize(
        ("written", "offsets", "expected"),
        [
            (
                "Time stops for nobody.  the ebb and flow of time",
                [[0, 4], [44, 48]],
                ["Time stops for nobody.", "the ebb and flow of time"],
            ),
            (
                "He ate a third of the pie.  Divided by two-thirds.",
                [[9, 14], [43, 48]],
                ["He ate a third of the pie.", "Divided by two-thirds."],
            ),
        ],
    )
    def test_splits_period_separated_examples_with_bold_evidence(
        self,
        attest: Callable[..., list[Sentence]],
        written: str,
        offsets: list[list[int]],
        expected: list[str],
    ) -> None:
        """
        Separate usage sentences retain their own located headword ranges.

        Args:
            attest: Runs extraction for example records.
            written: Combined example text.
            offsets: Bold ranges identifying each occurrence.
            expected: Separately retained examples.
        """
        headword = "Time" if written.startswith("Time") else "third"
        sentences = attest(
            {"text": written, "bold_text_offsets": offsets, "type": "example"},
            headword=headword,
        )

        assert [sentence.text for sentence in sentences] == expected
        assert all(sentence.word_offsets for sentence in sentences)

    def test_preserves_connected_sentences_with_single_spacing(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Ordinary punctuation does not prove two examples were combined.

        Args:
            attest: Runs extraction for example records.
        """
        written = "My English is weak. I wish my English were better."
        sentences = attest(
            {"text": written, "bold_text_offsets": [[3, 10], [30, 37]]},
            headword="English",
        )

        assert [sentence.text for sentence in sentences] == [written]

    def test_preserves_abbreviations_inside_one_example(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A short abbreviation does not delimit two examples.

        Args:
            attest: Runs extraction for example records.
        """
        written = "The bank met Dr. Bank at the bank."
        sentences = attest(
            {"text": written, "bold_text_offsets": [[4, 8], [21, 25]]},
        )

        assert [sentence.text for sentence in sentences] == [written]

    def test_preserves_continuing_alternatives_inside_one_example(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A continuation introduced by 'or' remains in its example.

        Args:
            attest: Runs extraction for example records.
        """
        written = "It looks like rain. or It looks like snow."
        first = written.index("looks")
        second = written.rindex("looks")
        sentences = attest(
            {
                "text": written,
                "bold_text_offsets": [[first, first + 5], [second, second + 5]],
            },
            headword="look",
        )

        assert [sentence.text for sentence in sentences] == [written]

    def test_excludes_audio_and_score_examples(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Score renderings and their captions do not illustrate word usage.
        """
        sentences = attest(
            {"text": "1"},
            {"text": r"{\key a \major a' b' cis d e fis gis a2}"},
            {"text": "The bank motif:\n{bes'a'cb'}"},
            {"text": "The bank is open."},
        )

        assert [sentence.text for sentence in sentences] == ["The bank is open."]

    def test_excludes_score_prefixes_from_source_markup(self) -> None:
        """
        Dump score provenance excludes a rendering even without LilyPond text.
        """
        sentence = clean_sentence(
            Attestation("The bank motif:\n7"),
            quoted=False,
            score_prefixes=("{{ux|en|The bank motif:<br>",),
        )

        assert sentence is None

    def test_excludes_editorial_examples(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Editorial links and labels are not usage examples.
        """
        sentences = attest(
            {"text": "Coordinate term: riverbank"},
            {"text": "Related terms (former names): old bank"},
            {"text": "Alternative forms: banke, banque"},
            {"text": "Synonym: riverbank"},
            {"text": "(initialisms)"},
            {"text": "The bank is open."},
        )

        assert [sentence.text for sentence in sentences] == ["The bank is open."]

    @pytest.mark.parametrize(
        ("written", "headword", "expected", "reference", "year"),
        [
            (
                "Jim Taylor (2000), DVD demystified, page 388: "
                + "“Because MiniDiscs are in a shell.”",
                "MiniDisc",
                "Because MiniDiscs are in a shell.",
                "Jim Taylor (2000), DVD demystified, page 388.",
                2000,
            ),
            (
                "Federal Writers Project (1952), West Virginia: A Guide to the "
                + "Mountain State, page 485: “Wellsburg thrived as a Gretna Green”",
                "Gretna Green",
                "Wellsburg thrived as a Gretna Green",
                "Federal Writers Project (1952), West Virginia: A Guide to the "
                + "Mountain State, page 485.",
                1952,
            ),
            (
                'William Morris, The Life and Death of Jason (1867), Book iv: "'
                + "I know a little garden-close",
                "garden",
                "I know a little garden-close",
                "William Morris, The Life and Death of Jason (1867), Book iv.",
                1867,
            ),
        ],
    )
    def test_recovers_inline_quotation_reference(
        self,
        attest: Callable[..., list[Sentence]],
        written: str,
        headword: str,
        expected: str,
        reference: str,
        year: int,
    ) -> None:
        """
        Inline source details become a reference without losing quoted text.

        Args:
            attest: Runs extraction for example records.
            written: Source citation followed by quoted content.
            headword: Word expected in the quoted content.
            expected: Quotation text after source removal.
            reference: Normalized source citation.
            year: Publication year parsed from the citation.
        """
        start = written.rfind(headword)
        sentences = attest(
            {"text": written, "bold_text_offsets": [[start, start + len(headword)]]},
            headword=headword,
        )

        (sentence,) = sentences

        assert isinstance(sentence, Quotation)
        assert sentence.text == expected
        assert sentence.reference == reference
        assert sentence.year == year
        assert (
            sentence.text[
                sentence.word_offsets[0].offset[0] : sentence.word_offsets[0].offset[1]
            ]
            == headword
        )

    def test_discards_standalone_bibliography(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Bibliographic shorthand without usage text is not an example.

        Args:
            attest: Runs extraction for example records.
        """
        sentences = attest(
            {"text": "〃, § 8.38.3, page 489"},
            {"text": "1823, in Crabb, Technol. Dict."},
            {"text": "The bank is open."},
        )

        assert [sentence.text for sentence in sentences] == ["The bank is open."]

    def test_removes_bibliography_identifiers_from_unquoted_examples(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Source identifiers delimit citations, while ambiguous titles are discarded.

        Args:
            attest: Runs extraction for example records.
        """
        sentences = attest(
            {
                "text": (
                    "An Albanologist wrote a study. "
                    "Jane Smith, John Doe, A Book, 2004, →ISBN."
                ),
                "bold_text_offsets": [[3, 15]],
            },
            {"text": "How to Brickle: A Book (1977, →ISBN"},
            headword="Albanologist",
        )

        assert [sentence.text for sentence in sentences] == [
            "An Albanologist wrote a study.",
        ]
        assert sentences[0].word_offsets[0].offset == (3, 15)

        dated = attest(
            {"text": "The dikkop is a bird. 1983 Birds of Africa. →ISBN."},
            headword="dikkop",
        )

        assert [sentence.text for sentence in dated] == ["The dikkop is a bird."]

    def test_recovers_quoted_usage_from_embedded_bibliography(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Publication identifiers delimit citations before and after usage text.
        """
        records = (
            {
                "text": '"Induration is hard." Jane Smith (1993), isbn 123.',
                "type": "example",
                "bold_text_offsets": [[1, 11]],
            },
            {
                "text": 'Jane Smith (2013), doi:10.1/example. "The nasoplasty worked."',
                "type": "example",
            },
            {
                "text": (
                    "The term has a meaning. It is used widely. "
                    "A Study of Words, John Doe (1962), doi:10.1/example"
                ),
                "type": "example",
            },
            {
                "text": (
                    "The rhizophagy works. (2013-C. Author, A Book, doi:10.1/example)"
                ),
                "type": "example",
            },
            {"text": "1989 (Spring), Jane Smith, A Book, doi:10.1/example"},
        )

        sentences = attest(*records, headword="induration")

        assert [sentence.text for sentence in sentences] == [
            "Induration is hard.",
            "The nasoplasty worked.",
            "The term has a meaning. It is used widely.",
            "The rhizophagy works.",
        ]
        assert sentences[0].text[slice(*sentences[0].word_offsets[0].offset)] == (
            "Induration"
        )

    def test_removes_multiline_bibliographic_headers(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Clear source headers leave usage text and dialogue intact.
        """
        sentences = attest(
            {"text": "18th century, Jane Smith, A Book\nThe term is used."},
            {"text": "H. G. Wells, A Book\nThe term occurs."},
            {"text": "Episode 12, A Series\nThe term appears."},
            {"text": "Are you ready?\nYes, the term is here."},
            headword="term",
        )

        assert [sentence.text for sentence in sentences] == [
            "The term is used.",
            "The term occurs.",
            "The term appears.",
            "Are you ready?\nYes, the term is here.",
        ]

    def test_preserves_lexical_mentions_of_identifiers(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A publication identifier in ordinary prose is not a citation.
        """
        sentences = attest(
            {"text": "In 2020, the book received an ISBN.", "type": "example"},
            {"text": "Doi Inthanon National Park is in Thailand.", "type": "example"},
            headword="ISBN",
        )

        assert [sentence.text for sentence in sentences] == [
            "In 2020, the book received an ISBN.",
            "Doi Inthanon National Park is in Thailand.",
        ]

    def test_preserves_ambiguous_layout_spacing(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        One unsupported em space remains ordinary sentence spacing.
        """
        written = "directionally correct \u2003 [= headed in the right direction]"

        sentences = attest({"text": written}, headword="directionally")

        assert [sentence.text for sentence in sentences] == [
            "directionally correct [= headed in the right direction]",
        ]

    def test_preserves_layout_inside_quotations(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Layout spacing never splits a quotation from its reference.
        """
        sentences = attest(
            {
                "text": "First bank.\u2003 Second bank.",
                "ref": "2000, A Source",
            },
        )

        assert [sentence.text for sentence in sentences] == [
            "First bank. Second bank.",
        ]

    @given(st.lists(raw_examples(), max_size=4))
    def test_classifies_unreferenced_examples(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
    ) -> None:
        """
        An editor wrote it, so there is no source to name.
        """
        sentences = attest(*examples)

        assert all(isinstance(sentence, Example) for sentence in sentences)
        assert [sentence.text for sentence in sentences] == [
            str(example["text"]).strip()
            for example in examples
            if any(character.isalpha() for character in str(example["text"]).strip())
        ]

    @given(st.data())
    def test_classifies_referenced_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        A reference is what makes a sentence evidence from somewhere.
        """
        year = data.draw(years)
        reference = data.draw(references(year))
        example = data.draw(
            raw_examples(
                texts=_LEXICAL_TEXTS,
                references=st.just(reference),
            ),
        )

        quotation = attest(example)[0]

        assert isinstance(quotation, Quotation)
        assert (quotation.text, quotation.reference) == (
            str(example["text"]).strip(),
            clean_reference(reference),
        )

    @pytest.mark.parametrize(
        ("written", "expected"),
        [
            ("(see title)", None),
            ("[See title.]", None),
            (
                'The term "suffrajitsu" came into common use. (also see title)',
                'The term "suffrajitsu" came into common use.',
            ),
            (
                "[see title] The term came into common use.",
                "The term came into common use.",
            ),
        ],
    )
    def test_removes_title_directives_from_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        written: str,
        expected: str | None,
    ) -> None:
        """
        A source-title directive is editorial text rather than quoted evidence.
        """
        sentences = attest({"text": written, "ref": "2015, A Title"})

        assert [sentence.text for sentence in sentences] == (
            [expected] if expected is not None else []
        )

    @given(st.lists(st.one_of(texts, blanks), max_size=4), st.data())
    def test_filters_blank_sentences(
        self,
        attest: Callable[..., list[Sentence]],
        written: list[str],
        data: st.DataObject,
    ) -> None:
        """
        A sentence of whitespace illustrates nothing.
        """
        examples = [data.draw(raw_examples(texts=st.just(text))) for text in written]

        assert [sentence.text for sentence in attest(*examples)] == [
            text.strip()
            for text in written
            if any(character.isalpha() for character in text)
        ]


class TestKinds:
    """
    Reading a sentence as the kind wiktextract says it is.
    """

    @given(
        suffix=st.text(alphabet=" :\t\u00a0", min_size=1),
        embedded=st.booleans(),
        headword=words,
    )
    def test_normalizes_reference_separators(
        self,
        attest: Callable[..., list[Sentence]],
        suffix: str,
        headword: str,
        *,
        embedded: bool,
    ) -> None:
        """
        Reference cleanup preserves internal colons, quotation dates, and offsets.
        """
        reference = "2000, A Title: A Subtitle, London: Publisher, https://example.org"
        written = reference + suffix
        raw: RawJson = (
            {"text": f"{written}\n{headword}", "type": "quotation"}
            if embedded
            else {"text": headword, "ref": written}
        )

        quotation = attest(raw, headword=headword)[0]

        assert isinstance(quotation, Quotation)
        assert quotation.reference == clean_reference(reference)
        assert quotation.year == 2000
        assert quotation.text == headword
        assert quotation.word_offsets == (
            WordOffset((0, len(headword)), (WordOffsetSource.LEMMATIZER,)),
        )
        assert attest(
            {"text": quotation.text, "ref": quotation.reference},
            headword=headword,
        ) == [quotation]

    def test_removes_an_unmatched_reference_bracket(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        An unmatched export delimiter does not become reference content.
        """
        quotation = attest(
            {
                "text": "A quoted sentence.",
                "ref": "[2015, An Author, A Title [revised]",
            }
        )[0]

        assert isinstance(quotation, Quotation)
        assert quotation.reference == "2015, An Author, A Title [revised]."

    @given(st.data())
    def test_preserves_explicit_examples(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        It read the markup; a reference is only what it left behind.
        """
        year = data.draw(years)
        reference = data.draw(references(year))
        text = data.draw(_LEXICAL_TEXTS)

        quotation = attest({"text": f"{reference}\n{text}", "type": "quotation"})[0]
        example = attest({"text": f"{reference}\n{text}", "type": "example"})[0]

        assert isinstance(quotation, Quotation)
        assert isinstance(example, Example)

    @given(_LEXICAL_TEXTS)
    def test_preserves_unreferenced_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        written: str,
    ) -> None:
        """
        An export tells the two apart by the reference, and it has none.
        """
        example = attest({"text": written, "type": "quotation"})[0]

        assert isinstance(example, Example)
        assert example.text == written.strip()

    @given(st.data())
    def test_recognizes_embedded_references(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        The source is in the text, which is why no reference came with it.
        """
        year = data.draw(years)
        reference = data.draw(references(year))
        text = data.draw(_LEXICAL_TEXTS)

        quotation = attest(
            {"text": f"{reference}\n{text}", "type": "quotation"},
        )[0]

        assert isinstance(quotation, Quotation)
        assert (quotation.text, quotation.reference, quotation.year) == (
            text.strip(),
            clean_reference(reference),
            year,
        )

    @given(reference=undated_references, sentence_text=_LEXICAL_TEXTS)
    @example(reference="''", sentence_text="a")
    @example(reference="A ", sentence_text="a")
    def test_interprets_undated_headers(
        self,
        attest: Callable[..., list[Sentence]],
        reference: str,
        sentence_text: str,
    ) -> None:
        """
        A break alone proves nothing: prose runs over lines too.
        """
        written = f"{reference}\n{sentence_text}"
        plain_header = re.sub(r"'{2,}", "", reference)
        plain_header = re.sub(r" {2,}", " ", plain_header)

        example = attest({"text": written})[0]

        assert isinstance(example, Example)
        assert example.text == f"{plain_header}\n{sentence_text}".strip()

        quotation = attest({"text": written, "type": "quotation"})[0]

        if cleaned_reference := clean_reference(reference):
            assert isinstance(quotation, Quotation)
            assert (quotation.text, quotation.reference) == (
                sentence_text.strip(),
                cleaned_reference,
            )
        else:
            assert isinstance(quotation, Example)
            assert quotation.text == sentence_text.strip()

    @given(st.data())
    def test_removes_embedded_bibliographic_lines(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        Explicit examples can still contain a source line before their text.
        """
        year = data.draw(years)
        written = f"{data.draw(references(year))}\n{data.draw(texts)}"

        example = attest({"text": written, "type": "example"})[0]

        assert isinstance(example, Example)
        assert example.text == written.partition("\n")[2].strip()

    @given(st.data())
    def test_excludes_bodyless_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        Splitting there would leave the reference, which attests nothing.
        """
        year = data.draw(years)
        written = f"{year}, A Book\n{data.draw(blanks)}"

        assert attest({"text": written, "type": "quotation"}) == []

    @given(st.data())
    def test_adjusts_reference_offsets(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        A headword named in a book title is not an occurrence of it.
        """
        headword = data.draw(words)
        year = data.draw(years)
        reference = f"{year}, {headword}, A Book"

        quotation = attest(
            {"text": f"{reference}\n{headword}", "type": "quotation"},
            headword=headword,
        )[0]

        assert quotation.text == headword
        assert quotation.word_offsets == (
            WordOffset(
                (0, len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )


class TestPointers:
    """
    Passing over what stands in for a sentence without being one.
    """

    def test_preserves_incomplete_pointer(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A navigation prefix without a target remains lexical text.
        """
        written = "For quotations SEE"

        sentences = attest({"text": written}, headword="see")

        assert [sentence.text for sentence in sentences] == [written]

    @given(words)
    def test_excludes_citation_pointers(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
    ) -> None:
        """
        It navigates somewhere; it attests nothing.
        """
        assert attest({"text": _build_pointer(headword)}, headword=headword) == []

    @given(words, sentence_kinds)
    def test_excludes_typed_pointers(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        kind: str,
    ) -> None:
        """
        An example type cannot turn a citation link into lexical evidence.
        """
        written = _build_pointer(headword)

        sentences = attest({"text": written, "type": kind}, headword=headword)

        assert sentences == []

    @given(words, st.data())
    def test_preserves_pointer_prefixes(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """
        Pointer detection requires the complete template.
        """
        written = f"For quotations {data.draw(texts)}"

        sentences = attest({"text": written}, headword=headword)

        expected = normalize_formatting(Attestation(written)).text

        assert [sentence.text for sentence in sentences] == [expected]


class TestYears:
    """
    Reading a year off a reference, and filtering on it.
    """

    @given(st.data())
    def test_extracts_reference_year(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        A reference is prose, so the year is taken where it is recognised.
        """
        year = data.draw(years)
        example = data.draw(
            raw_examples(
                texts=_LEXICAL_TEXTS,
                references=references(year),
            ),
        )

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
    def test_parses_wiktionary_dates(
        self,
        attest: Callable[..., list[Sentence]],
        reference: str,
        expected: int | None,
    ) -> None:
        """
        The shapes are Wiktionary's own, and the year opens each of them.
        """
        quotation = attest({"text": "He runs.", "ref": reference})[0]

        assert isinstance(quotation, Quotation)
        assert quotation.year == expected

    @given(st.lists(years, max_size=4), st.none() | years, st.none() | years, st.data())
    def test_filters_quotation_years(
        self,
        attest: Callable[..., list[Sentence]],
        dated: list[int],
        minimum_year: int | None,
        maximum_year: int | None,
        data: st.DataObject,
    ) -> None:
        """
        Both bounds are inclusive, and either stands on its own.
        """
        examples = [
            data.draw(
                raw_examples(
                    texts=_LEXICAL_TEXTS,
                    references=references(year),
                ),
            )
            for year in dated
        ]

        sentences = attest(
            *examples,
            minimum_year=minimum_year,
            maximum_year=maximum_year,
        )

        assert [
            sentence.year for sentence in sentences if isinstance(sentence, Quotation)
        ] == [
            year
            for year in dated
            if (minimum_year is None or year >= minimum_year)
            and (maximum_year is None or year <= maximum_year)
        ]

    @given(st.lists(raw_examples(), max_size=4), st.none() | years, st.data())
    def test_bounds_reduce_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
        minimum_year: int | None,
        data: st.DataObject,
    ) -> None:
        """
        A bound is asked for to leave something out, never to let something in.
        """
        dated_examples = [
            data.draw(raw_examples(references=references(data.draw(years))))
            for _ in examples
        ]

        unfiltered_sentences = attest(*examples, *dated_examples)
        filtered_sentences = attest(
            *examples, *dated_examples, minimum_year=minimum_year
        )

        assert set(filtered_sentences) <= set(unfiltered_sentences)

    @given(st.data())
    def test_excludes_undated_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        A quotation nothing can date cannot be shown to be inside a bound.
        """
        undated = data.draw(raw_examples(references=undated_references))

        assert attest(undated, minimum_year=data.draw(years)) == []

    @given(st.data())
    def test_preserves_undated_quotations(
        self,
        attest: Callable[..., list[Sentence]],
        data: st.DataObject,
    ) -> None:
        """
        An undated quotation only stands in the way once a bound is set.
        """
        reference = data.draw(undated_references)
        undated = data.draw(
            raw_examples(
                texts=_LEXICAL_TEXTS,
                references=st.just(reference),
            ),
        )

        quotation = attest(undated)[0]

        assert isinstance(quotation, Quotation)
        assert (quotation.reference, quotation.year) == (
            clean_reference(reference),
            None,
        )

    @given(st.lists(raw_examples(), max_size=3), st.data())
    def test_preserves_unfiltered_examples(
        self,
        attest: Callable[..., list[Sentence]],
        examples: list[RawJson],
        data: st.DataObject,
    ) -> None:
        """
        The bounds reach quotations alone, an example carrying no date.
        """
        year = data.draw(years)
        excluded_quotation = data.draw(raw_examples(references=references(year)))

        assert attest(*examples, excluded_quotation, minimum_year=year + 1) == attest(
            *examples
        )

    @given(st.data())
    def test_preserves_unattested_senses(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """
        A sense is what its gloss says, whatever evidence the bounds leave it.
        """
        year = data.draw(years)
        example = data.draw(raw_examples(references=references(year)))
        entry = data.draw(
            raw_entries(
                senses=st.lists(
                    raw_senses(examples=st.just([example])),
                    min_size=1,
                    max_size=1,
                ),
            ),
        )

        lemmas = extract([entry], minimum_year=year + 1)

        assert [sense.sentences for sense in lemmas[0].senses] == [[]]


class TestWordOffsets:
    """
    Where the lemma occurs in the sentences attesting it.
    """

    @given(words, st.data())
    def test_locates_sentence_lemma(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """
        The headword is a form of itself, so it is looked for like the rest.
        """
        example = data.draw(raw_examples(texts=st.just(f"1 {headword} 2")))

        assert attest(example, headword=headword)[0].word_offsets == (
            WordOffset(
                (2, 2 + len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )

    def test_merges_offset_sources(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Agreement remains distinguishable from either method alone.
        """
        sentences = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[2, 6]],
            },
        )

        assert sentences[0].word_offsets == (
            WordOffset(
                (2, 6),
                (
                    WordOffsetSource.BOLD,
                    WordOffsetSource.LEMMATIZER,
                ),
            ),
        )

    def test_preserves_offset_disagreements(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        A later review needs both proposals where the methods disagree.
        """
        sentences = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[7, 14]],
            },
        )

        assert sentences[0].word_offsets == (
            WordOffset((2, 6), (WordOffsetSource.LEMMATIZER,)),
            WordOffset((7, 14), (WordOffsetSource.BOLD,)),
        )

    def test_discards_invalid_offsets(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        An invalid source range cannot identify text for review.
        """
        sentences = attest(
            {
                "text": "a bank account",
                "bold_text_offsets": [[2, 100]],
            },
        )

        assert sentences[0].word_offsets == (
            WordOffset((2, 6), (WordOffsetSource.LEMMATIZER,)),
        )

    def test_shifts_bold_offsets(
        self,
        attest: Callable[..., list[Sentence]],
    ) -> None:
        """
        Bold ranges remain relative to the exported sentence.
        """
        reference = "2026, bank"
        start = len(reference) + 1
        sentences = attest(
            {
                "text": f"{reference}\nbank",
                "type": "quotation",
                "bold_text_offsets": [[start, start + 4]],
            },
        )

        assert sentences[0].word_offsets == (
            WordOffset(
                (0, 4),
                (
                    WordOffsetSource.BOLD,
                    WordOffsetSource.LEMMATIZER,
                ),
            ),
        )

    @given(words, words, st.data())
    def test_locates_listed_inflections(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        inflection: str,
        data: st.DataObject,
    ) -> None:
        """
        A sentence attests the lemma in whatever form it needs.
        """
        form = data.draw(raw_forms(forms=_pad_word(inflection)))
        example = data.draw(raw_examples(texts=st.just(f"1 {inflection} 2")))

        sentences = attest(example, headword=headword, forms=[form])

        assert (2, 2 + len(inflection)) in (
            word_offset.offset for word_offset in sentences[0].word_offsets
        )

    @given(words, _SERVICE_TAGS, st.data())
    def test_excludes_service_forms(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        tag: str,
        data: st.DataObject,
    ) -> None:
        """
        An inflection table names itself, its template and its transliterations.
        """
        service_form = data.draw(
            words.filter(lambda form: form.casefold() != headword.casefold()),
        )

        form = data.draw(raw_forms(forms=st.just(service_form), tags=st.just([tag])))
        example = data.draw(raw_examples(texts=st.just(f"1 {service_form} 2")))

        sentences = attest(example, headword=headword, forms=[form])

        assert (2, 2 + len(service_form)) not in (
            word_offset.offset for word_offset in sentences[0].word_offsets
        )

    @given(words, _EMPTY_CELLS, st.data())
    def test_excludes_empty_forms(
        self,
        attest: Callable[..., list[Sentence]],
        headword: str,
        cell: str,
        data: st.DataObject,
    ) -> None:
        """
        A dash stands for a form that does not exist, and blank for none at all.
        """
        form = data.draw(raw_forms(forms=st.just(cell)))
        example = data.draw(raw_examples(texts=st.just(f"1 {headword} - 2")))

        sentences = attest(example, headword=headword, forms=[form])

        assert sentences[0].word_offsets == (
            WordOffset(
                (2, 2 + len(headword)),
                (WordOffsetSource.LEMMATIZER,),
            ),
        )


class TestVariants:
    """
    How else a lemma is spelled, which one thing says.
    """

    @given(
        words,
        st.lists(words, min_size=1, max_size=3).map(".".join),
        parts_of_speech,
        st.data(),
    )
    def test_collects_spelling_variants(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        pos: POS,
        data: st.DataObject,
    ) -> None:
        """
        Another spelling sits on a page of its own and points back.
        """
        variant_entry: RawJson = {
            "word": spelling,
            "pos": pos.value,
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Alternative spelling of {headword}."],
                    "tags": ["alt-of"],
                    "alt_of": [{"word": headword}],
                },
            ],
        }
        defining_entry = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just(pos.value)),
        )

        lemmas = extract([variant_entry, defining_entry])

        assert [lemma.variants for lemma in lemmas] == [
            frozenset({spelling}) - {headword},
        ]

    @given(words, words, st.data())
    def test_collects_name_variants(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """
        Proper-noun spellings use Wiktextract's name code.
        """
        variant_entry: RawJson = {
            "word": spelling,
            "pos": "name",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Alternative spelling of {headword}."],
                    "tags": ["alt-of"],
                    "alt_of": [{"word": headword}],
                },
            ],
        }
        defining_entry = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just("name")),
        )

        lemmas = extract([variant_entry, defining_entry])

        assert [lemma.variants for lemma in lemmas] == [
            frozenset({spelling}) - {headword},
        ]

    @given(words, words, st.data())
    def test_excludes_inflected_variants(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """
        Plural forms are excluded from spelling variants.
        """
        inflected_entry: RawJson = {
            "word": spelling,
            "pos": "noun",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Plural of {headword}."],
                    "tags": ["form-of", "plural"],
                    "form_of": [{"word": headword}],
                },
            ],
        }
        defining_entry = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just("noun")),
        )

        lemmas = extract([inflected_entry, defining_entry])

        assert [lemma.variants for lemma in lemmas] == [frozenset()]

    @given(words, words, st.data())
    def test_excludes_listed_variants(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """
        An Alternative forms section names derivations as readily as spellings.
        """
        form = data.draw(
            raw_forms(forms=st.just(spelling), tags=st.just(["alternative"])),
        )
        entry = data.draw(
            raw_entries(headwords=st.just(headword), forms=st.just([form])),
        )

        (lemma,) = extract([entry])

        assert lemma.variants == frozenset()

    @given(words, words, st.data())
    def test_filters_variant_categories(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        spelling: str,
        data: st.DataObject,
    ) -> None:
        """
        A spelling of the noun says nothing about how the verb is written.
        """
        variant_entry: RawJson = {
            "word": spelling,
            "pos": "verb",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": [f"Alternative spelling of {headword}."],
                    "tags": ["alt-of"],
                    "alt_of": [{"word": headword}],
                },
            ],
        }
        defining_entry = data.draw(
            raw_entries(headwords=st.just(headword), pos_codes=st.just("noun")),
        )

        lemmas = extract([variant_entry, defining_entry])

        assert [lemma.variants for lemma in lemmas] == [frozenset()]


class TestTranslations:
    """
    What other languages call the entry, which Wiktionary hangs off the entry.
    """

    @given(
        words,
        languages,
        _TRANSLATION_GLOSSES,
        st.data(),
    )
    def test_groups_translation_glosses(
        self,
        extract: Callable[..., list[Lemma]],
        translation: str,
        language: str,
        gloss: str,
        data: st.DataObject,
    ) -> None:
        """
        The gloss is stripped, an editor having written it by hand.
        """
        translation_record = data.draw(
            raw_translations(
                translations=st.just(translation),
                codes=st.just(language),
                glosses=st.just(gloss),
            ),
        )
        entry = data.draw(raw_entries(translations=st.just([translation_record])))

        (lemma,) = extract([entry])

        assert lemma.translation_tables[0].gloss == normalize_statement(gloss)
        assert lemma.translation_tables[0].translations == {
            Language(language): frozenset({translation}),
        }

    @given(
        words,
        words,
        languages,
        _TRANSLATION_GLOSSES,
        st.data(),
    )
    def test_groups_translation_languages(
        self,
        extract: Callable[..., list[Lemma]],
        first: str,
        second: str,
        language: str,
        gloss: str,
        data: st.DataObject,
    ) -> None:
        """
        One meaning is often said more than one way in the same language.
        """
        translation_records = [
            data.draw(
                raw_translations(
                    translations=st.just(word),
                    codes=st.just(language),
                    glosses=st.just(gloss),
                ),
            )
            for word in (first, second)
        ]
        entry = data.draw(raw_entries(translations=st.just(translation_records)))

        (lemma,) = extract([entry])

        table = next(
            table
            for table in lemma.translation_tables
            if table.gloss == normalize_statement(gloss)
        )

        assert table.translations[Language(language)] == frozenset({first, second})

    def test_preserves_wiktextract_language_label(
        self,
        extract: Callable[..., list[Lemma]],
    ) -> None:
        """
        Collected tables retain the language name without changing word groups.

        Args:
            extract: Runs extraction for raw Wiktextract entries.
        """
        (lemma,) = extract(
            [
                {
                    "word": "bank",
                    "pos": "noun",
                    "lang_code": LANGUAGE,
                    "senses": [{"glosses": ["A financial institution."]}],
                    "translations": [
                        {
                            "word": "banca",
                            "lang_code": "it",
                            "lang": "Italian",
                            "sense": "A financial institution.",
                        },
                    ],
                },
                {
                    "word": "bank",
                    "pos": "noun",
                    "lang_code": LANGUAGE,
                    "senses": [{"glosses": ["A financial institution."]}],
                    "translations": [
                        {
                            "word": "istituto di credito",
                            "lang_code": "it",
                            "sense": "A financial institution.",
                        },
                    ],
                },
            ],
        )

        (table,) = lemma.translation_tables

        assert table.translations == {
            Language("it", "Italian"): frozenset({"banca", "istituto di credito"}),
        }

    @given(
        words,
        words,
        _TRANSLATION_GLOSSES,
        st.data(),
    )
    def test_removes_gloss_references(
        self,
        extract: Callable[..., list[Lemma]],
        first: str,
        second: str,
        gloss: str,
        data: st.DataObject,
    ) -> None:
        """
        A displayed see-also reference does not become part of the table gloss.
        """
        referenced_gloss = f"{gloss} — see also alternative\ncleanup note"
        translation_records = [
            data.draw(
                raw_translations(
                    translations=st.just(translation),
                    codes=st.just(language),
                    glosses=st.just(table_gloss),
                ),
            )
            for translation, language, table_gloss in (
                (first, "it", gloss),
                (second, "fr", referenced_gloss),
            )
        ]
        entry = data.draw(raw_entries(translations=st.just(translation_records)))

        (lemma,) = extract([entry])

        assert len(lemma.translation_tables) == 1

        table = lemma.translation_tables[0]

        assert table.gloss == normalize_statement(gloss)
        assert table.translations == {
            Language("it"): frozenset({first}),
            Language("fr"): frozenset({second}),
        }

    @given(st.sampled_from(("word", "lang_code", "sense")), st.data())
    def test_excludes_incomplete_translations(
        self,
        extract: Callable[..., list[Lemma]],
        key: str,
        data: st.DataObject,
    ) -> None:
        """
        A translation is filed under its gloss and its language, or nowhere.
        """
        translation_record = data.draw(raw_translations())
        translation_record[key] = data.draw(blanks)

        entry = data.draw(raw_entries(translations=st.just([translation_record])))

        (lemma,) = extract([entry])

        assert not lemma.translation_tables

    @given(st.data())
    def test_preserves_empty_translations(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """
        Wiktionary writes a translation table for a fraction of its entries.
        """
        entry = data.draw(raw_entries(translations=st.just([])))

        (lemma,) = extract([entry])

        assert not lemma.translation_tables


class TestSynonyms:
    """
    Other words standing for what a sense means.
    """

    @given(words, words, st.data())
    def test_associates_sense_synonyms(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        synonym: str,
        data: st.DataObject,
    ) -> None:
        """
        A synonym under a sense stands for that meaning alone.
        """
        synonym_record = data.draw(raw_synonyms(words=st.just(synonym)))
        sense = data.draw(raw_senses(synonyms=st.just([synonym_record])))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense])),
        )

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == (() if synonym == headword else (synonym,))

    @given(words, words, words, st.data())
    def test_preserves_synonym_order(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        first: str,
        second: str,
        data: st.DataObject,
    ) -> None:
        """
        A sense may be put more than one way, and order is what is read.
        """
        synonym_records = [
            data.draw(raw_synonyms(words=st.just(word))) for word in (first, second)
        ]
        sense = data.draw(raw_senses(synonyms=st.just(synonym_records)))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense])),
        )

        (lemma,) = extract([entry])

        # Generated words may coincide; deduplication preserves their first occurrence.
        expected: dict[str, None] = {}

        for word in (first, second):
            if word != headword:
                expected[word] = None

        assert lemma.senses[0].synonyms == tuple(expected)

    @given(words, st.data())
    def test_excludes_headword_synonyms(
        self,
        extract: Callable[..., list[Lemma]],
        headword: str,
        data: st.DataObject,
    ) -> None:
        """
        A word is not offered as another way to say itself.
        """
        synonym_record = data.draw(raw_synonyms(words=st.just(headword)))
        sense = data.draw(raw_senses(synonyms=st.just([synonym_record])))
        entry = data.draw(
            raw_entries(headwords=st.just(headword), senses=st.just([sense])),
        )

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ()

    @given(st.data())
    def test_preserves_empty_synonyms(
        self,
        extract: Callable[..., list[Lemma]],
        data: st.DataObject,
    ) -> None:
        """
        Most senses offer no synonym at all.
        """
        sense = data.draw(raw_senses(synonyms=st.just([])))
        entry = data.draw(raw_entries(senses=st.just([sense])))

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ()

    @pytest.mark.parametrize("synonym", [",", ".", ";"])
    def test_discards_punctuation_synonyms(
        self,
        extract: Callable[..., list[Lemma]],
        synonym: str,
    ) -> None:
        """
        Punctuation alone does not identify a synonym.
        """
        entry: RawJson = {
            "word": "word",
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["A meaning."], "synonyms": [{"word": synonym}]}],
        }

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ()

    @pytest.mark.parametrize(
        "broken",
        [
            "Vulgar:",
            "shmekl (q",
            "Yiddish)",
            "interstate[:w:Interstate Highway System|Interstate Highway System]]>",
        ],
    )
    def test_discards_malformed_synonyms(
        self,
        extract: Callable[..., list[Lemma]],
        broken: str,
    ) -> None:
        """
        Parser fragments must not enter the synonym set.
        """
        entry: RawJson = {
            "word": "word",
            "pos": "noun",
            "lang_code": "en",
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "synonyms": [
                        {"word": broken},
                        {"word": "make a move (on)"},
                    ],
                },
            ],
        }

        (lemma,) = extract([entry])

        assert lemma.senses[0].synonyms == ("make a move (on)",)


class TestStrayReferences:
    """
    Passing over a reference left standing where a sentence belongs.
    """

    @given(years)
    def test_excludes_stray_references(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
    ) -> None:
        """
        A citation on one line names a source and attests nothing.
        """
        assert attest({"text": f"{year}, John Milton, A Book"}) == []

    @given(years)
    def test_preserves_date_prefixes(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
    ) -> None:
        """
        A year is how a sentence about a year opens, and it is a sentence.
        """
        written = f"{year} saw the release of many great films."

        assert attest({"text": written})[0].text == written

    @given(years, st.data())
    def test_recognizes_quotation_body(
        self,
        attest: Callable[..., list[Sentence]],
        year: int,
        data: st.DataObject,
    ) -> None:
        """
        A line break separates the reference from its sentence.
        """
        written = f"{year}, A Book\n{data.draw(_LEXICAL_TEXTS)}"

        assert attest({"text": written, "type": "quotation"})[0].text
