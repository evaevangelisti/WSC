"""
Exercise audited text policies through extraction and offset-preserving cleanup.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from kwic import Locator
from strategies import RawJson, words

from wsc.extract import WiktionaryExtractor
from wsc.extract.translations import clean_translations
from wsc.extract.wiktionary.parts.glosses import clean_gloss
from wsc.extract.wiktionary.parts.sentences import clean_sentence
from wsc.models import (
    Attestation,
    Language,
    Lemma,
    TranslationTable,
    WordOffset,
    WordOffsetSource,
)


@pytest.fixture
def extract(
    workspace: Callable[[], Path],
    write_entries: Callable[[Path, Iterable[RawJson]], Path],
    locator: Locator,
) -> Callable[[RawJson, tuple[TranslationTable, ...]], list[Lemma]]:
    """
    Run the public extractor with both raw and supplementary source text.
    """

    def run(
        fields: RawJson,
        supplementary: tuple[TranslationTable, ...] = (),
    ) -> list[Lemma]:
        """
        Collect a defining entry with the supplied fields.
        """
        entry: RawJson = {
            "word": "sample entry",
            "pos": "noun",
            "lang_code": "en",
            "senses": [{"glosses": ["A meaning."]}],
            **fields,
        }
        source = write_entries(workspace() / "source.jsonl", [entry])
        extractor = WiktionaryExtractor(
            None,
            None,
            None,
            locator,
            {"sample_entry.noun": supplementary},
        )

        return list(extractor.extract(source))

    return run


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        (
            "A meaning (see other). A second definition.",
            "A meaning. A second definition.",
        ),
        (
            "A meaning. See other. A second definition.",
            "A meaning. A second definition.",
        ),
        ("A function (continuous; see Usage notes).", "A function (continuous)."),
        ("A town. See Town on Wikipedia.Wikipedia", "A town."),
        ("See done, for: Punished because of.", "Punished because of."),
        ("See. vide(te)", "See. vide(te)."),
        ("1108 Demeter, a main belt asteroid.", "1108 Demeter, a main belt asteroid."),
        ("1599 Geneva Bible", "1599 Geneva Bible."),
        ("1000 lambdas", "1000 lambdas."),
        ("Compare", "Compare."),
        ("See; see also.", "See; see also."),
        ("A hierarchy heading:", "A hierarchy heading."),
        (
            "A rodent (but see also its synonyms), native to America.",
            "A rodent, native to America.",
        ),
        (
            "Rights. see: Wikipedia:Reciprocity (international relations).",
            "Rights.",
        ),
        ("A definition (For further discussion, see the appendix).", "A definition."),
        (
            "A definition (more formally, see the chart), with a qualification.",
            "A definition, with a qualification.",
        ),
        (
            "A definition (said of a treatment, see treatment).",
            "A definition (said of a treatment).",
        ),
        (
            "A definition (cf. another), with a qualification.",
            "A definition, with a qualification.",
        ),
        (
            "A definition. Compare fly (verb (regular)) and line (verb).",
            "A definition.",
        ),
        ("A piece of fabric cf. gusset.", "A piece of fabric."),
        ("Real Madrid CF.", "Real Madrid CF."),
        (
            "To buy a ticket for a movie, see it and then go into another movie.",
            "To buy a ticket for a movie, see it and then go into another movie.",
        ),
        ("A bishop's see: merely titular.", "A bishop's see: merely titular."),
        (
            "To compare two objects; to assess their differences.",
            "To compare two objects; to assess their differences.",
        ),
        (
            "A failure to perceive something (see, hear, feel, etc.).",
            "A failure to perceive something (see, hear, feel, etc.).",
        ),
        (
            "To see by foresight; see clairvoyantly; view telepathically.",
            "To see by foresight; see clairvoyantly; view telepathically.",
        ),
        ("A tax code (see other).", "A tax code."),
        (
            "The planting of ivy (see Ivy Day (United States), etc.",
            "The planting of ivy.",
        ),
        ("A meaning.\nSee also: other", "A meaning."),
        ("unknown", "Unknown."),
        ("A [[leaf|leafy]] plant.", "A leafy plant."),
        (
            "The formula C<sub>2</sub> and 10<sup>15</sup>.",
            "The formula C_{2} and 10^{15}.",
        ),
        ("The set #92;mathbb#123;Z#125;.", r"The set \mathbb{Z}."),
        (
            "x \u2208 \u211d and x\u00b2 \u2265 \u22121.",
            r"x \in \mathbb{R} and x^{2} \geq -1.",
        ),
        (
            "\U0001d465 + \U0001d466 = \U0001d467.",
            r"\mathit{x} + \mathit{y} = \mathit{z}.",
        ),
        ("A meaning. https://example.org/source", "A meaning."),
        ("A meaning. https://example.org/a, https://example.org/b.", "A meaning."),
        ("A meaning https://example.org/a, https://example.org/b.", "A meaning."),
        (
            "A meaning (Especially after still. For more on this use, see https://example.org/a)",
            "A meaning (Especially after still.)",
        ),
        ("An [[unclosed link", None),
        ("Template:init of", None),
        ("2013, Neils Axt, ChronoTales, page 15", None),
        (r"The relation \forallx\existsy.", None),
        (
            r"The relation x \in \mathbb{Z}.",
            r"The relation x \in \mathbb{Z}.",
        ),
        ("The product x⋅y.", r"The product x\cdot y."),
    ],
)
def test_cleans_definitions(
    extract: Callable[..., list[Lemma]],
    written: str,
    expected: str | None,
) -> None:
    """
    Definitions retain meaning while damaged leaves cannot broaden to parents.
    """
    result = extract({"senses": [{"glosses": ["A parent.", written]}]})

    if expected is None:
        assert result == []
    else:
        assert result[0].senses[0].glosses == ("A parent.", expected)


@pytest.mark.parametrize(
    "written",
    [
        "See other.",
        "* see: The Warren.",
        "For additional senses, see the individual entries.",
    ],
)
def test_removes_navigation_levels(
    extract: Callable[..., list[Lemma]],
    written: str,
) -> None:
    """
    A navigation-only hierarchy level does not erase its defining parent.
    """
    result = extract({"senses": [{"glosses": ["A meaning.", written]}]})

    assert result[0].senses[0].glosses == ("A meaning.",)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Near-synonyms: sample, example", None),
        ("Near synonym: sample", None),
        ("Meronyms: handle", None),
        ("Holonym: hand", None),
        ("Usage notes: see other", None),
        ("See Citations:sample", None),
        ("See also quotations under sample.", None),
        ("Also see: United National Congress, Trinidad and Tobago", None),
        ("A sample entry. Also see: another entry.", "A sample entry."),
        ("A sample entry (also see: another entry).", "A sample entry."),
        ("For examples using this term, see Citations:sample.", None),
        ("See you at five.", "See you at five."),
        ("I left; see you tomorrow.", "I left; see you tomorrow."),
        ("He wrote (see you soon).", "He wrote (see you soon)."),
        ("A line\n.\nAnother line", "A line\n.\nAnother line"),
        (
            "In HTML, <b>bold</b> tags mark text.",
            "In HTML, <b>bold</b> tags mark text.",
        ),
        (
            "Use the title() filter: <h2>{{ post.title|title }}</h2>",
            "Use the title() filter: <h2>{{ post.title|title }}</h2>",
        ),
        ("A '''sample''' entry.", "A sample entry."),
        ("ParmÃ©nide â€“ a sample entry.", "Parménide – a sample entry."),
        ("A sam\u00adple\u2060 entry.", "A sample entry."),
        (
            "A #92;mathbb#123;Z#125; sample entry.",
            r"A \mathbb{Z} sample entry.",
        ),
        ("A #92;forallx sample entry.", None),
        ("A {{unexpanded|sample}} entry.", None),
        (". Compare caducous.", None),
        (
            "The predicand denotes he (cf. he was downhearted).",
            "The predicand denotes he (cf. he was downhearted).",
        ),
    ],
)
def test_cleans_sentences(
    extract: Callable[..., list[Lemma]],
    written: str,
    expected: str | None,
) -> None:
    """
    Typed examples still exclude metadata and preserve literal and multiline text.
    """
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "examples": [{"text": written, "type": "example"}],
                }
            ]
        }
    )

    assert [sentence.text for sentence in result[0].senses[0].sentences] == (
        [expected] if expected is not None else []
    )


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        (
            "The sample entry remains. https://example.org/source",
            "The sample entry remains.",
        ),
        (
            "The sample entry remains. — Author, 2007 https://example.org/source",
            "The sample entry remains.",
        ),
        (
            "The sample entry remains. (A Book, 2007https://example.org/source",
            "The sample entry remains.",
        ),
        (
            "https://example.org/source The sample entry remains.",
            "The sample entry remains.",
        ),
        (
            "Aldrichimica Acta Volume 30 No 4 (pdf) from Sigma-Aldrich\n"
            + "The sample entry remains.",
            "The sample entry remains.",
        ),
    ],
)
def test_removes_unstructured_example_bibliography(
    extract: Callable[..., list[Lemma]],
    written: str,
    expected: str,
) -> None:
    """
    Unreferenced source details leave example text and word offsets intact.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
        written: Example text containing source details.
        expected: Example text after source removal.
    """
    start = written.index("sample entry")
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "examples": [
                        {
                            "text": written,
                            "bold_text_offsets": [[start, start + len("sample entry")]],
                        },
                    ],
                },
            ],
        },
    )

    (sentence,) = result[0].senses[0].sentences

    assert sentence.text == expected
    assert any(
        sentence.text[offset.offset[0] : offset.offset[1]] == "sample entry"
        for offset in sentence.word_offsets
    )


def test_preserves_urls_used_as_example_text(
    extract: Callable[..., list[Lemma]],
) -> None:
    """
    An explicit example may demonstrate a URL rather than cite a source.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
    """
    written = "The sample entry links to https://example.org/."
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "examples": [{"text": written, "type": "example"}],
                },
            ],
        },
    )

    assert result[0].senses[0].sentences[0].text == written


def test_discards_bibliography_without_example_text(
    extract: Callable[..., list[Lemma]],
) -> None:
    """
    A standalone author, title, journal, and URL do not illustrate a word.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
    """
    written = (
        'Pemberton, S. George and Robert W. Frey 1991. "A sample entry". '
        + "Ichnos, 1: 317–325. https://example.org/source"
    )
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "examples": [{"text": written}],
                },
            ],
        },
    )

    assert result[0].senses[0].sentences == []


@pytest.mark.parametrize(
    "written",
    [
        "See also the life around you.",
        "Also see: the life around you.",
        "We said (see you at five) and left.",
        "The address is https://example.org/.",
        "A chemical 2-[[1-amino]-2-oxo] compound.",
        "She said, ‘hello’.\nAnother line.",
    ],
)
def test_preserves_quoted_content(
    extract: Callable[..., list[Lemma]],
    written: str,
) -> None:
    """
    Genuine quotations keep navigation-like prose, URLs, and chemical brackets.
    """
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning."],
                    "examples": [{"text": written, "ref": "2000, A Book"}],
                }
            ]
        }
    )

    assert result[0].senses[0].sentences[0].text == written


@given(
    token=words,
    reference=st.sampled_from(
        [
            "See also quotation under cyclopian.",
            "See also quotations under vapory.",
            "See Citations:sample.",
            "Also see: quotation under cyclopian.",
            "See also: quotations under vapory.",
            "But see: Citations:sample.",
        ],
    ),
    position=st.sampled_from(["before", "between", "after"]),
)
@example(token="S", reference="See also quotation under cyclopian.", position="after")
def test_removes_separate_editorial_lines(
    token: str,
    reference: str,
    position: str,
) -> None:
    """
    Removing editorial lines preserves quoted prose and exact token provenance.
    """
    lines = [f"A {token}.", f"Another {token}."]
    lines.insert({"before": 0, "between": 1, "after": 2}[position], reference)
    written = "\n".join(lines)
    start = written.index(f"Another {token}.") + len("Another ")
    reference_start = written.index(reference)
    sources = (WordOffsetSource.BOLD, WordOffsetSource.LEMMATIZER)
    value = Attestation(
        written,
        word_offsets=(
            WordOffset((start, start + len(token)), sources),
            WordOffset((reference_start, reference_start + 3), sources),
        ),
    )

    cleaned = clean_sentence(value, quoted=True)
    expected = f"A {token}.\nAnother {token}."
    final_start = expected.rindex(token)

    assert cleaned is not None
    assert cleaned == Attestation(
        expected,
        word_offsets=(WordOffset((final_start, final_start + len(token)), sources),),
    )
    assert clean_sentence(cleaned, quoted=True) == cleaned


@given(
    prefix=st.sampled_from(["Also see", "also SEE", "BUT see", "See also"]),
    spacing=st.sampled_from(["", " ", "\t", "  "]),
    bullet=st.sampled_from(["", "* ", "• ", "- "]),
)
def test_excludes_navigation_across_fields(
    extract: Callable[..., list[Lemma]],
    prefix: str,
    spacing: str,
    bullet: str,
) -> None:
    """
    Colon navigation is excluded consistently across both translation sources.
    """
    written = f"{bullet}{prefix}{spacing}:{spacing}another entry"
    result = extract(
        {
            "senses": [
                {
                    "glosses": ["A meaning.", written],
                    "examples": [{"text": written, "type": "example"}],
                },
            ],
            "translations": [
                {"sense": written, "lang_code": "fr", "word": "mot"},
                {"sense": "A meaning.", "lang_code": "fr", "word": written},
            ],
        },
        (TranslationTable("old", written, {Language("fr"): frozenset({"mot"})}),),
    )

    assert result[0].senses[0].glosses == ("A meaning.",)
    assert not result[0].senses[0].sentences
    assert not result[0].translation_tables


@pytest.mark.parametrize(
    ("language", "written", "expected"),
    [
        ("fi", "see laskettu aika", None),
        ("de", "but see Kuchen", None),
        ("de", "but see", None),
        ("fr", "also see feuille", None),
        ("en", "see: snowflake and softie", None),
        ("de", "see adjektivisches Demonstrativpronomen", None),
        ("de", "See Thesaurus:Heidelbeere", None),
        ("de", "See von Galiläa", ("de", "See von Galiläa")),
        ("af", "See van Japan", ("af", "See van Japan")),
        ("fy", "See fan Azov", ("fy", "See fan Azov")),
        ("sco", "See o Japan", ("sco", "See o Japan")),
        ("et", "see", ("et", "see")),
        ("en", "translation", ("en", "translation")),
        ("cmn", "see entry) 可惜", ("cmn", "可惜")),
        ("fi", "poltto (see polttomoottori)", ("fi", "poltto")),
        (
            "fi",
            "hyvät ja huonot ajat (but also see myötä- ja vastoinkäymiset)",
            ("fi", "hyvät ja huonot ajat"),
        ),
        ("fi", "ilman (jotakin)", ("fi", "ilman")),
        ("sq", "lirë (i/e)", ("sq", "lirë")),
        ("ko", "자유적(自由的)이다", ("ko", "자유적이다")),
        ("cy", "{{t|1=cy|2=post|3=m}}", ("cy", "post")),
        ("en", "{{t+|cmn|極簡主義|tr=jíjiǎn zhǔyì}}", ("cmn", "極簡主義")),
        ("fr", "[[feuille|feuilles]]", ("fr", "feuilles")),
        ("fr", "[[feuille", None),
        ("yue", "{{|yue|洛陽}}", None),
        ("cy", "{{t|cy|post}} junk }}", None),
        ("el", "\u2060", None),
        ("nan-tws", "[script needed]", None),
        ("es", "[es el", None),
        ("sms", "määnpââ\N{ACUTE ACCENT}jj", ("sms", "määnpââ\N{ACUTE ACCENT}jj")),
        ("ml", "അ\u200dആ", ("ml", "അ\u200dആ")),
        (" fa-ira ", " واژه ", ("fa-ira", "واژه")),
        ("nds-de", "Woord", ("nds-de", "Woord")),
        ("bad|code", "word", None),
    ],
)
def test_cleans_both_translation_sources(
    extract: Callable[..., list[Lemma]],
    language: str,
    written: str,
    expected: tuple[str, str] | None,
) -> None:
    """
    Raw and supplementary translations follow the same field-specific policy.
    """
    heading = "A '''meaning''' (see other)."
    supplementary = (
        TranslationTable("old", heading, {Language(language): frozenset({written})}),
    )
    records = [{"sense": heading, "lang_code": language, "word": written}]
    raw = extract({"translations": records})[0]
    merged = extract({}, supplementary)[0]

    assert raw.translation_tables == merged.translation_tables

    if expected is None:
        assert not raw.translation_tables
    else:
        code, word = expected
        assert raw.translation_tables[0].gloss == "A meaning."
        assert raw.translation_tables[0].translations == {
            Language(code): frozenset({word}),
        }


@pytest.mark.parametrize(
    "reference",
    [
        "otherwise see [[mercury#Translations",
        "otherwise see [[mercury#Translations|mercury]]",
        "otherwise see also mercury",
        "see also mercury",
    ],
)
def test_removes_editorial_translation_tails(
    extract: Callable[..., list[Lemma]],
    reference: str,
) -> None:
    """
    An incomplete editorial link does not erase its complete defining heading.
    """
    heading = "cognate translations of hydrargyrum"
    written = f"{heading} — {reference}"
    supplementary = (
        TranslationTable(
            "old",
            written,
            {Language("la"): frozenset({"hydrargyrum"})},
        ),
    )
    records = [{"sense": written, "lang_code": "la", "word": "hydrargyrum"}]
    raw = extract({"translations": records})[0]
    merged = extract({}, supplementary)[0]

    assert raw.translation_tables == merged.translation_tables
    assert raw.translation_tables[0].gloss == "Cognate translations of hydrargyrum."


def test_normalizes_translation_gloss_punctuation(
    extract: Callable[..., list[Lemma]],
) -> None:
    """
    Translation headings use the same terminal punctuation as sense glosses.
    """
    result = extract(
        {
            "translations": [
                {
                    "sense": "A hierarchical meaning:",
                    "lang_code": "it",
                    "word": "parola",
                }
            ]
        }
    )

    assert result[0].translation_tables[0].gloss == "A hierarchical meaning."


@pytest.mark.parametrize(
    "heading",
    [
        "translation",
        "translations",
        "translations  to be checked",
        "translation gloss",
        "sense",
    ],
)
def test_excludes_supplementary_placeholders(
    extract: Callable[..., list[Lemma]],
    heading: str,
) -> None:
    """
    A supplementary table must carry a definition before semantic alignment.
    """
    result = extract(
        {},
        (
            TranslationTable(
                "old",
                heading,
                {Language("it"): frozenset({"parola"})},
            ),
        ),
    )

    assert result[0].translation_tables == ()


@given(
    token=words,
    prefix=st.lists(
        st.sampled_from(["\u00ad", "\u2060", "'''", "&amp;", "x "]), max_size=8
    ),
    quoted=st.booleans(),
)
@example(token="A", prefix=["'''", "'''", "\u00ad", "'''", "'''"], quoted=False)
def test_relocates_existing_ranges(
    token: str,
    prefix: list[str],
    *,
    quoted: bool,
) -> None:
    """
    Exact substitutions keep repeated tokens and all offset provenance aligned.
    """
    written = "".join(prefix) + " " + token + " " + token
    start = len(written) - len(token)
    value = Attestation(
        written,
        word_offsets=(
            WordOffset(
                (start, len(written)),
                (WordOffsetSource.BOLD, WordOffsetSource.LEMMATIZER),
            ),
        ),
    )

    cleaned = clean_sentence(value, quoted=quoted)

    assert cleaned is not None
    assert len(cleaned.word_offsets) == 1
    offset = cleaned.word_offsets[0]
    assert cleaned.text[offset.offset[0] : offset.offset[1]] == token
    assert offset.offset[1] == len(cleaned.text)
    assert offset.sources == value.word_offsets[0].sources
    assert clean_sentence(cleaned, quoted=quoted) == cleaned


@given(st.text(max_size=200))
def test_handles_arbitrary_unicode(
    written: str,
) -> None:
    """
    Malformed input remains deterministic and never produces invalid ranges.
    """
    value = Attestation(
        written,
        word_offsets=(WordOffset((0, len(written)), (WordOffsetSource.BOLD,)),)
        if written
        else (),
    )
    cleaned = clean_sentence(value, quoted=True)
    gloss = clean_gloss(written)
    translation = clean_translations("und", written)

    if cleaned is not None:
        assert cleaned.text
        assert all(
            0 <= item.offset[0] < item.offset[1] <= len(cleaned.text)
            for item in cleaned.word_offsets
        )
        assert clean_sentence(cleaned, quoted=True) == cleaned

    if gloss:
        assert clean_gloss(gloss) == gloss

    if translation is not None:
        language, alternatives = translation

        assert all(
            clean_translations(language, alternative)
            == (language, frozenset({alternative}))
            for alternative in alternatives
        )


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("lirë (i/e)", frozenset({"lirë"})),
        ("word;/term,", frozenset({"word", "term"})),
        ("el/la/lo más", frozenset({"el más", "la más", "lo más"})),
        ("nascer/pôr do sol", frozenset({"nascer do sol", "pôr do sol"})),
        ("uJanuwari class 1a/2a", frozenset({"uJanuwari"})),
        ("Abtrünniger m/Abtrünnige", frozenset({"Abtrünniger", "Abtrünnige"})),
        ("Wort \\ Begriff", frozenset({"Wort", "Begriff"})),
        ("armado [con]", frozenset({"armado"})),
        ("[el] ala", frozenset({"ala"})),
        ("\U0001d400", frozenset({r"\mathbf{A}"})),
        ("[nocą]", frozenset({"nocą"})),
        ("funcionario[a]", frozenset({"funcionario", "funcionaria"})),
        ("[anglicism]skipping", frozenset({"skipping"})),
        (
            "لِيرَة f or لَيْرَة",
            frozenset({"لِيرَة", "لَيْرَة"}),
        ),
        ("papallona de la c blanca", frozenset({"papallona de la c blanca"})),
        (
            "kapzsi/telhetetlen ember/lény",
            frozenset(
                {
                    "kapzsi ember",
                    "kapzsi lény",
                    "telhetetlen ember",
                    "telhetetlen lény",
                },
            ),
        ),
    ],
)
def test_keeps_lexical_translation_alternatives(
    written: str,
    expected: frozenset[str],
) -> None:
    """
    Translation cleanup removes metadata without losing complete variants.
    """
    assert clean_translations("und", written) == ("und", expected)


@pytest.mark.parametrize(
    ("language", "written", "expected"),
    [
        ("la", "Abdēra n pl and", "Abdēra"),
        ("fr", "les îles Ioniennes f plural", "les îles Ioniennes"),
        ("wal", "Inggilizettuwa sg or", "Inggilizettuwa"),
        ("sv", "grönt uncountable", "grönt"),
        ("ru", "сзыва́ть impf", "сзыва́ть"),
        ("la", "Āram gender unattested", "Āram"),
        ("la", "Āram gender unknown", "Āram"),
        ("de", "plural", "plural"),
        ("pt", "primeira pessoa do singular", "primeira pessoa do singular"),
        ("uk", "приско́рювати impf прискорити", "приско́рювати impf прискорити"),
        ("pl", "możliwe, że", "możliwe, że"),
        ("kab", "Iwunak Yeddukklen n Temrikt", "Iwunak Yeddukklen n Temrikt"),
    ],
)
def test_removes_unambiguous_translation_grammar(
    extract: Callable[..., list[Lemma]],
    language: str,
    written: str,
    expected: str,
) -> None:
    """
    Suffix metadata is removed without splitting lexical or ambiguous text.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
        language: Translation language code.
        written: Raw translation with possible grammar notes.
        expected: Translation retained after cleanup.
    """
    result = extract(
        {
            "translations": [
                {"sense": "A meaning.", "lang_code": language, "word": written},
            ],
        },
    )

    assert result[0].translation_tables[0].translations == {
        Language(language): frozenset({expected}),
    }


@pytest.mark.parametrize(
    ("language", "written", "expected"),
    [
        ("fr", "ton m ta f tes", frozenset({"ton", "ta", "tes"})),
        ("it", "tuo m tua f tuoi m pl tue", frozenset({"tuo", "tua", "tuoi", "tue"})),
        ("de", "Musikant m Musikantin", frozenset({"Musikant", "Musikantin"})),
        (
            "de",
            "unerlaubte Entfernung f unerlaubte Entfernung von der Truppe",
            frozenset(
                {
                    "unerlaubte Entfernung",
                    "unerlaubte Entfernung von der Truppe",
                },
            ),
        ),
        ("zh", "詞典 /词典", frozenset({"詞典", "词典"})),
        ("he", "שחור \\ שָׁחֹר", frozenset({"שחור", "שָׁחֹר"})),
        ("de", "Wort m / Begriff", frozenset({"Wort", "Begriff"})),
    ],
)
def test_separates_explicit_translation_alternatives(
    extract: Callable[..., list[Lemma]],
    language: str,
    written: str,
    expected: frozenset[str],
) -> None:
    """
    Repeated gender labels and spaced slashes delimit complete forms.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
        language: Translation language code.
        written: Raw translation with explicit alternative boundaries.
        expected: Complete lexical alternatives.
    """
    result = extract(
        {
            "translations": [
                {"sense": "A meaning.", "lang_code": language, "word": written},
            ],
        },
    )

    assert result[0].translation_tables[0].translations == {
        Language(language): expected,
    }


@pytest.mark.parametrize(
    ("language", "written", "expected"),
    [
        (
            "fr",
            "Cayes, Les Cayes, Aux Cayes",
            frozenset({"Cayes", "Les Cayes", "Aux Cayes"}),
        ),
        (
            "eo",
            "fanarioto, fanariano, fenerano",
            frozenset({"fanarioto", "fanariano", "fenerano"}),
        ),
        (
            "de",
            "grußloser Abschied, heimliche Abfahrt etc.",
            frozenset({"grußloser Abschied", "heimliche Abfahrt"}),
        ),
        ("pl", "możliwe, że", frozenset({"możliwe, że"})),
    ],
)
def test_separates_supported_comma_translation_lists(
    extract: Callable[..., list[Lemma]],
    language: str,
    written: str,
    expected: frozenset[str],
) -> None:
    """
    Distinct translations are separated without breaking lexical commas.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
        language: Translation language code.
        written: Raw translation containing a possible comma list.
        expected: Lexical alternatives retained after cleanup.
    """
    result = extract(
        {
            "translations": [
                {"sense": "A meaning.", "lang_code": language, "word": written}
            ]
        },
    )

    assert result[0].translation_tables[0].translations == {
        Language(language): expected,
    }


@pytest.mark.parametrize(
    "written",
    [
        "often mistakenly called förkörsrätt",
        "Thesaurus:isoisä",
        "see Thesaurus:käppiä",
    ],
)
def test_discards_editorial_translation_labels(
    extract: Callable[..., list[Lemma]],
    written: str,
) -> None:
    """
    An editorial pointer cannot serve as a translated lemma.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
        written: Editorial translation text.
    """
    result = extract(
        {"translations": [{"sense": "A meaning.", "lang_code": "sv", "word": written}]},
    )

    assert result[0].translation_tables == ()


def test_discards_ambiguous_middle_gender(
    extract: Callable[..., list[Lemma]],
) -> None:
    """
    A lone internal gender label cannot safely identify both forms.

    Args:
        extract: Public extractor for a supplied Wiktextract entry.
    """
    result = extract(
        {
            "translations": [
                {
                    "sense": "A meaning.",
                    "lang_code": "de",
                    "word": "Wort m Begriff",
                },
            ],
        },
    )

    assert result[0].translation_tables == ()


@given(st.lists(words, min_size=2, max_size=5))
def test_uses_underscores_only_in_identifiers(
    extract: Callable[..., list[Lemma]],
    fragments: list[str],
) -> None:
    """
    Every child identifier shares the normalized headword without changing text.
    """
    headword = " ".join(fragments) + "-suffix"
    fields: RawJson = {
        "word": headword,
        "translations": [{"sense": "A meaning.", "lang_code": "it", "word": "parola"}],
        "senses": [{"glosses": ["A meaning."], "examples": [{"text": headword}]}],
    }

    (lemma,) = extract(fields)

    assert lemma.lemma == headword
    assert lemma.id == "_".join(fragments) + "-suffix.noun"
    assert lemma.senses[0].id.startswith(lemma.id + ".")
    assert lemma.translation_tables[0].id.startswith(lemma.id + ".tr.")
    assert lemma.senses[0].sentences[0].text == headword
