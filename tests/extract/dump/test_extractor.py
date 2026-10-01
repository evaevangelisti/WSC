"""
Exercise off-page translations through complete source documents.
"""

import bz2
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from documents import dump, page
from hypothesis import given
from hypothesis import strategies as st
from strategies import words

from wsc.extract import (
    DumpExtractor,
    build_off_page_translations,
    index_translation_glosses,
    read_off_page_translations,
    write_off_page_translations,
)
from wsc.extract.wiktionary.schema import RawEntry
from wsc.identifiers import translation_table_id
from wsc.models import POS, Language, TranslationTable


@given(
    translations=st.lists(words, min_size=1, max_size=15),
    template=st.sampled_from(
        ("t", "t+", "tt", "tt+", "t-check", "t+check", "t-simple"),
    ),
    compressed=st.booleans(),
)
def test_respects_translation_boundaries(
    workspace: Callable[[], Path],
    translations: list[str],
    template: str,
    *,
    compressed: bool,
) -> None:
    """
    Duplicate translations merge without leaking across sections or tables.
    """
    markup = "\n".join(
        [
            "==French==",
            "===Noun===",
            "{{trans-top|ignored}}",
            "{{t|it|foreign}}",
            "==English==",
            "===Noun===",
            "{{t|it|outside}}",
            "{{trans-top|'''meaning'''|id=meaning}}",
            *(f"{{{{{template}|it|{word}|m}}}}" for word in translations),
            "{{t|it}} {{t||}}",
            "{{trans-bottom}}",
            "{{t|it|after}}",
            "===Verb===",
            "{{t|it|without a table}}",
            "{{trans-top-also|action|act|do}}",
            "{{t|fr|agir}}",
            "{{trans-bottom}}",
            "==German==",
            "===Noun===",
            "{{trans-top|foreign}}",
            "{{t|it|wrong language}}",
        ],
    )
    document = dump(
        page("entry/translations", markup),
        page("Talk:entry/translations", markup, namespace=1),
        page("ordinary", markup),
        page("empty/translations", ""),
    ).encode()
    source = workspace() / ("dump.xml.bz2" if compressed else "dump.xml")
    _ = source.write_bytes(bz2.compress(document) if compressed else document)

    records = {
        record.pos: record for record in DumpExtractor("English").extract(source)
    }

    assert set(records) == {POS.NOUN, POS.VERB}

    noun = records[POS.NOUN]

    assert noun.lemma == "entry"
    assert not noun.pointers
    assert len(noun.translations) == 1
    assert noun.translations[0].id == translation_table_id("entry.noun", "Meaning.")
    assert noun.translations[0].translations == {
        Language("it"): frozenset(translations),
    }
    assert records[POS.VERB].translations[0].translations == {
        Language("fr"): frozenset({"agir"}),
    }
    assert not records[POS.VERB].pointers


def test_keeps_original_math_in_supplemental_table_gloss(
    tmp_path: Path,
) -> None:
    """
    A table read directly from Wikitext retains the formula's source.
    """
    source = tmp_path / "dump.xml"
    _ = source.write_text(
        dump(
            page(
                "ratio/translations",
                "==English==\n===Noun===\n"
                + "{{trans-top|A ratio of <math>\\left(\\frac{1}{2}\\right)</math>}}\n"
                + "{{t|fr|moitié}}\n{{trans-bottom}}",
            ),
        ),
        encoding="utf-8",
    )

    (entry,) = DumpExtractor("English").extract(source)

    assert [table.gloss for table in entry.translations] == [
        r"A ratio of \left(\frac{1}{2}\right).",
    ]


def test_does_not_duplicate_a_rendered_math_table() -> None:
    """
    Source TeX and its Wiktextract rendering identify the same table.
    """
    markup = (
        "==English==\n===Noun===\n"
        + "{{trans-top|A ratio of <math>\\left(\\frac{1}{2}\\right)</math>}}\n"
        + "{{t|fr|moitié}}\n{{trans-bottom}}"
    )
    parsed_glosses = {
        "ratio.noun": frozenset({"a ratio of (1/2)"}),
    }

    pages = DumpExtractor("English").extract_pages(
        [("ratio", markup)],
        parsed_glosses,
    )

    assert list(pages) == []


@given(
    translations=st.lists(words, min_size=1, max_size=15),
    template=st.sampled_from(("trans-see", "trans-top-see")),
)
def test_resolves_translation_pointers(
    workspace: Callable[[], Path],
    translations: list[str],
    template: str,
) -> None:
    """
    Pointers resolve by part of speech and retain stable destination identities.
    """
    directory = workspace()
    source = directory / "dump.xml"
    _ = source.write_text(
        dump(
            page(
                "entry",
                "==English==\n===Noun===\n"
                + f"{{{{{template}|Meaning.|target|missing}}}}\n"
                + "{{trans-see|target}}\n"
                + "{{trans-see|spaced gloss|target}}\n"
                + "{{trans-see|unmatched|target}}\n"
                + "{{trans-see|}}",
            ),
            page(
                "entry/translations",
                "==English==\n===Noun===\n{{trans-top|Meaning.}}\n"
                + "{{t|fr|mot}}\n{{trans-bottom}}",
            ),
            page(
                "target/translations",
                "==English==\n===Noun===\n"
                + "{{trans-top|MEANING. — see also alternative}}\n"
                + "{{t|de|Wort}}\n{{trans-bottom}}",
            ),
        ),
        encoding="utf-8",
    )
    entries: list[RawEntry] = [
        {
            "word": "target",
            "lang_code": "en",
            "pos": "noun",
            "translations": [
                {
                    "word": word,
                    "lang_code": "it",
                    "lang": "Italian",
                    "sense": "meaning — see also alternative",
                },
            ],
        }
        for word in translations
    ]
    entries.extend(
        [
            {"word": "target", "lang_code": "en", "pos": "unknown"},
            {
                "word": "target",
                "lang_code": "en",
                "pos": "noun",
                "translations": [
                    {"word": "single", "lang_code": "it", "sense": "target"},
                    {"word": "excluded", "lang_code": "it", "sense": "other"},
                    {
                        "word": "spacing",
                        "lang_code": "it",
                        "sense": "spaced   gloss.",
                    },
                ],
            },
            {
                "word": "target",
                "lang_code": "en",
                "pos": "verb",
                "translations": [
                    {"word": "excluded verb", "lang_code": "it", "sense": "meaning"},
                ],
            },
            {
                "word": "target",
                "lang_code": "mpt",
                "pos": "noun",
                "translations": [
                    {
                        "word": "intrusa",
                        "lang_code": "it",
                        "sense": "meaning",
                    },
                ],
            },
        ],
    )
    result = build_off_page_translations(
        DumpExtractor("English").extract(source),
        entries,
    )
    output = directory / "translations.json"
    write_off_page_translations(output, result)

    assert read_off_page_translations(output) == result
    assert set(result) == {"entry.noun", "target.noun"}

    tables = {table.gloss: table for table in result["entry.noun"]}

    assert set(tables) == {"Meaning.", "Target.", "Spaced gloss."}
    assert tables["Meaning."].translations == {
        Language("it", "Italian"): frozenset(translations),
        Language("fr"): frozenset({"mot"}),
        Language("de"): frozenset({"Wort"}),
    }
    assert tables["Target."].translations == {
        Language("it", "Italian"): frozenset({"single"}),
    }
    assert tables["Spaced gloss."].translations == {
        Language("it", "Italian"): frozenset({"spacing"}),
    }
    assert all(
        table.id == translation_table_id("entry.noun", gloss)
        for gloss, table in tables.items()
    )


def test_reads_code_indexed_supplemental_cache(
    workspace: Callable[[], Path],
) -> None:
    """
    A cache created before language keys remains usable for collection.

    Args:
        workspace: Sets aside a directory for the cache file.
    """
    path = workspace() / "translations.json"
    _ = path.write_text(
        json.dumps(
            {
                "entry.noun": [
                    {
                        "id": "t",
                        "gloss": "Meaning.",
                        "translations": {"it": ["banca"]},
                        "languages": [{"code": "it", "label": "Italian"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert read_off_page_translations(path) == {
        "entry.noun": (
            TranslationTable(
                "t",
                "Meaning.",
                {Language("it", "Italian"): frozenset({"banca"})},
            ),
        ),
    }


def test_resolves_name_pointers(
    workspace: Callable[[], Path],
) -> None:
    """
    Proper-noun pointers match Wiktextract's name code.
    """
    source = workspace() / "dump.xml"
    _ = source.write_text(
        dump(
            page(
                "entry",
                "==English==\n===Proper noun===\n" + "{{trans-see|meaning|target}}",
            ),
        ),
        encoding="utf-8",
    )
    entries: list[RawEntry] = [
        {
            "word": "target",
            "lang_code": "en",
            "pos": "name",
            "translations": [
                {
                    "word": "nome",
                    "lang_code": "it",
                    "sense": "meaning",
                },
            ],
        },
    ]

    result = build_off_page_translations(
        DumpExtractor("English").extract(source),
        entries,
    )

    assert result["entry.propn"][0].translations == {
        Language("it"): frozenset({"nome"}),
    }


def test_fills_missing_english_tables(
    workspace: Callable[[], Path],
) -> None:
    """
    Raw markup supplies only tables absent from the parsed entry.
    """
    source = workspace() / "dump.xml"
    _ = source.write_text(
        dump(
            page(
                "entry",
                "==English==\n===Noun===\n"
                + "{{trans-top|known}}\n{{t|it|nota}}\n{{trans-bottom}}\n"
                + "{{trans-top|missing}}\n{{t|fr|mot}}\n{{trans-bottom}}",
            ),
        ),
        encoding="utf-8",
    )
    entries: list[RawEntry] = [
        {
            "word": "entry",
            "lang_code": "en",
            "pos": "noun",
            "translations": [
                {
                    "word": "parola",
                    "lang_code": "it",
                    "sense": "known",
                },
            ],
        },
        {
            "word": "entry",
            "lang_code": "mpt",
            "pos": "noun",
            "translations": [
                {
                    "word": "mot",
                    "lang_code": "fr",
                    "sense": "missing",
                },
            ],
        },
    ]
    parsed_glosses = index_translation_glosses(entries)

    result = build_off_page_translations(
        DumpExtractor("English").extract(source, parsed_glosses),
        entries,
    )

    assert [table.gloss for table in result["entry.noun"]] == ["Missing."]
    assert result["entry.noun"][0].translations == {
        Language("fr"): frozenset({"mot"}),
    }


@pytest.mark.parametrize("level", [3, 4])
@pytest.mark.parametrize("boundary", ["Pronoun", "Etymology 2", "language"])
def test_isolates_translation_sections(
    tmp_path: Path,
    level: int,
    boundary: str,
) -> None:
    """
    Section changes close unfinished tables and release the previous POS.
    """
    heading = "=" * level
    boundary_heading = "=" * (3 if boundary == "Etymology 2" else level)
    transition = (
        ["==French==", f"{heading}Noun{heading}", "==English=="]
        if boundary == "language"
        else [f"{boundary_heading}{boundary}{boundary_heading}"]
    )
    markup = "\n".join(
        [
            "==English==",
            f"{heading}Noun{heading}",
            "{{trans-top|kept}}",
            "{{t|it|prima}}",
            *transition,
            "{{t|it|leaked}}",
            "{{trans-top|excluded}}",
            "{{t|it|estranea}}",
            "{{trans-see|excluded|target}}",
            f"{heading}Noun{heading}",
            "{{t|it|still outside}}",
            "{{trans-top|retained}}",
            "{{t|it|seconda}}",
            "{{trans-bottom}}",
        ],
    )
    source = tmp_path / "dump.xml"
    _ = source.write_text(
        dump(page("entry/translations", markup)),
        encoding="utf-8",
    )

    (record,) = DumpExtractor("English").extract(source)

    assert record.pos == POS.NOUN
    assert not record.pointers
    assert {table.gloss: table.translations for table in record.translations} == {
        "Kept.": {Language("it"): frozenset({"prima"})},
        "Retained.": {Language("it"): frozenset({"seconda"})},
    }


@given(
    first=words,
    second=words,
    numbered=st.booleans(),
    multiline=st.booleans(),
)
def test_reads_nested_arguments_in_document_order(
    workspace: Callable[[], Path],
    first: str,
    second: str,
    *,
    numbered: bool,
    multiline: bool,
) -> None:
    """
    Piped links, numbered parameters, and same-line table boundaries retain words.
    """
    separator = "\n" if multiline else ""
    heading = f"To produce [[leaf|{first}]] and 10<sup>15</sup>"
    arguments = f"1=fr|2=[[leaf|{second}]]" if numbered else f"fr|[[leaf|{second}]]"
    markup = (
        "==English==\n===Noun===\n"
        + f"{{{{trans-top|{separator}{heading}}}}}"
        + f"{{{{t|{separator}{arguments}|note={{{{q|rare|dated}}}}}}}}"
        + "{{trans-bottom}}{{t|it|outside}}"
        + "\n{{trans-top|translations  to be checked}}{{t|it|placeholder}}"
        + "{{trans-bottom}}"
        + "\n<!-- {{trans-top|comment}}{{t|it|comment}} -->"
        + "<nowiki>{{trans-top|literal}}{{t|it|literal}}</nowiki>"
    )
    source = workspace() / "dump.xml"
    _ = source.write_text(dump(page("sample entry/translations", markup)))

    (record,) = DumpExtractor("English").extract(source)
    (table,) = record.translations

    assert table.gloss == f"To produce {first} and 10^{{15}}."
    assert table.id.startswith("sample_entry.noun.tr.")
    assert table.translations == {Language("fr"): frozenset({second})}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("1=it|fr|2=parola", "fr"),
        ("fr|1=it|2=parola", "it"),
    ],
)
def test_uses_last_parameter_assignment(
    workspace: Callable[[], Path],
    arguments: str,
    expected: str,
) -> None:
    """
    Explicit numbering and implicit positions obey source-order assignment.
    """
    markup = (
        "==English==\n===Noun===\n{{trans-top|meaning}}"
        + f"{{{{t|{arguments}}}}}"
        + "{{trans-bottom}}"
    )
    source = workspace() / "dump.xml"
    _ = source.write_text(dump(page("entry/translations", markup)))

    (record,) = DumpExtractor("English").extract(source)

    assert record.translations[0].translations == {
        Language(expected): frozenset({"parola"}),
    }


def test_recovers_partially_damaged_parsed_tables(
    workspace: Callable[[], Path],
) -> None:
    """
    A usable word cannot hide a truncated translation from source recovery.
    """
    entries: list[RawEntry] = [
        {
            "word": "leaf",
            "pos": "noun",
            "lang_code": "en",
            "translations": [
                {"sense": "plant part", "lang_code": "fr", "word": "feuille"},
                {"sense": "plant part", "lang_code": "it", "word": "[[foglio"},
            ],
        },
    ]
    markup = (
        "==English==\n===Noun===\n{{trans-top|plant part}}\n"
        + "{{t|fr|feuille}} {{t|it|[[foglio|foglia]]}}\n{{trans-bottom}}"
    )
    source = workspace() / "dump.xml"
    _ = source.write_text(dump(page("leaf", markup)))

    (record,) = DumpExtractor("English").extract(
        source, index_translation_glosses(entries)
    )

    assert record.translations[0].translations == {
        Language("fr"): frozenset({"feuille"}),
        Language("it"): frozenset({"foglia"}),
    }
