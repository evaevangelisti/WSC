"""Exercise off-page translations through complete source documents."""

import bz2
from collections.abc import Callable
from pathlib import Path

from documents import dump, page
from hypothesis import given
from hypothesis import strategies as st
from strategies import words

from wsc.extract import (
    DumpExtractor,
    build_off_page_translations,
    read_off_page_translations,
    write_off_page_translations,
)
from wsc.extract.wiktionary.schema import RawEntry
from wsc.identifiers import translation_table_id
from wsc.models import POS


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
    """Duplicate translations merge without leaking across sections or tables."""
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
            "{{trans-top-also|action}}",
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
    assert noun.translations[0].id == translation_table_id("entry.noun", "meaning")
    assert noun.translations[0].translations == {"it": frozenset(translations)}
    assert records[POS.VERB].translations[0].translations == {"fr": frozenset({"agir"})}


@given(
    translations=st.lists(words, min_size=1, max_size=15),
    template=st.sampled_from(("trans-see", "trans-top-see")),
)
def test_resolves_translation_pointers(
    workspace: Callable[[], Path],
    translations: list[str],
    template: str,
) -> None:
    """Pointers resolve by part of speech and retain stable destination identities."""
    directory = workspace()
    source = directory / "dump.xml"
    _ = source.write_text(
        dump(
            page(
                "entry",
                "==English==\n===Noun===\n"
                + f"{{{{{template}|meaning|target|missing}}}}\n"
                + "{{trans-see|target}}\n{{trans-see|}}",
            ),
            page(
                "entry/translations",
                "==English==\n===Noun===\n{{trans-top|meaning}}\n"
                + "{{t|fr|mot}}\n{{trans-bottom}}",
            ),
        ),
        encoding="utf-8",
    )
    entries: list[RawEntry] = [
        {
            "word": "target",
            "pos": "noun",
            "translations": [{"word": word, "lang_code": "it", "sense": "other"}],
        }
        for word in translations
    ]
    entries.extend(
        [
            {"word": "target", "pos": "unknown"},
            {
                "word": "target",
                "pos": "verb",
                "translations": [{"word": "excluded", "lang_code": "it"}],
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
    assert set(result) == {"entry.noun"}

    tables = {table.gloss: table for table in result["entry.noun"]}

    assert set(tables) == {"meaning", "target"}
    assert tables["meaning"].translations == {
        "it": frozenset(translations),
        "fr": frozenset({"mot"}),
    }
    assert tables["target"].translations == {"it": frozenset(translations)}
    assert all(
        table.id == translation_table_id("entry.noun", gloss)
        for gloss, table in tables.items()
    )
