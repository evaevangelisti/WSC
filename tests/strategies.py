"""
Generators the properties are drawn from.

A generator says what a source may write rather than what it usually writes,
so the alphabets reach past ASCII and the lists reach down to empty.
"""

import string
from datetime import date

from hypothesis import strategies as st

from wsc.models import (
    POS,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    WordOffset,
    WordOffsetSource,
)

type RawJson = dict[str, object]
"""One decoded JSON object, as wiktextract writes them."""

# Letters as far as Latin Extended-B, whose case Python's re folds the way
# str does.
_LETTERS = st.characters(categories=("Ll", "Lu"), max_codepoint=0x24F)

# How a reference names a date, the year aside. Wiktionary writes one in more
# ways than these, and the year opens each.
_REFERENCE_SHAPES = [
    "{year}, A Book",
    "{year}s, A Song",
    "c. {year}, A Play",
    "{year} August 11, A Newspaper",
]

# Prose holding no digit, so that no year is read off it.
_UNDATED = string.ascii_letters + " ,.'"

words = st.text(alphabet=_LETTERS, min_size=1, max_size=8)
"""One written form, whether a headword or an inflection of one."""

glosses = st.text(min_size=1, max_size=40).filter(lambda gloss: bool(gloss.strip()))
"""What a sense says it means, in whatever an editor wrote it."""

texts = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="\n"),
    min_size=1,
    max_size=60,
).filter(lambda text: bool(text.strip()))
"""The sentence an example or a quotation carries, on one line: a break is
where a quotation carrying its own source divides the two."""

sentence_kinds = st.sampled_from(["example", "quotation"])
"""What wiktextract calls a sentence, where it says which kind it read."""

form_tags = st.sampled_from(["form-of", "alt-of"])
"""A tag marking a sense that inflects a headword rather than defining it."""

blanks = st.text(alphabet=" \t\n", min_size=1, max_size=3)
"""What reads as nothing at all once it has been stripped."""

years = st.integers(min_value=1000, max_value=2099)
"""A year a reference may name, from the first century of printing to this one."""

undated_references = st.text(alphabet=_UNDATED, min_size=1, max_size=20).filter(
    lambda reference: bool(reference.strip())
)
"""A source naming no year, which is a quotation nothing can date."""

languages = st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=3)
"""Wiktionary's code for one edition."""

wordnet_versions = st.integers(min_value=1000, max_value=9999).map(str)
"""The year one edition of the wordnet came out."""

dump_dates = st.dates(
    min_value=date(2001, 1, 1),
    max_value=date(2099, 12, 31),
).map(lambda day: day.strftime("%Y%m%d"))
"""The day a dump began, as the directory holding it is named."""

parts_of_speech: st.SearchStrategy[POS] = st.sampled_from(POS)
"""One part of speech the collector keeps."""

etymologies = st.integers(min_value=1, max_value=9).map(str)
"""Which etymology of a page an entry sits under, as Wiktionary numbers them."""

pos_codes = parts_of_speech.map(lambda pos: pos.value)
"""Wiktextract's code for one of them."""

unknown_pos_codes = st.text(
    alphabet=string.ascii_lowercase,
    min_size=1,
    max_size=6,
).filter(lambda code: code not in {pos.value for pos in POS})
"""A part of speech Wiktionary describes and the collector does not keep."""

# What a builder falls back on when a test has nothing to say about that
# part of an entry, drawn rather than left empty.

_OFFSETS = st.integers(min_value=0, max_value=60)

_WORD_OFFSET_SOURCES = st.lists(
    st.sampled_from(WordOffsetSource),
    min_size=1,
    max_size=len(WordOffsetSource),
    unique=True,
).map(tuple)

_WORD_OFFSETS = st.lists(
    st.builds(
        WordOffset,
        st.tuples(_OFFSETS, _OFFSETS),
        _WORD_OFFSET_SOURCES,
    ),
    max_size=3,
).map(tuple)

_UNQUOTED: st.SearchStrategy[str | None] = st.none()

_UNTYPED: st.SearchStrategy[str | None] = st.none()

_GLOSS_CHAINS = st.lists(glosses, min_size=1, max_size=3)

_LABELS = st.lists(words, max_size=2)

_ENGLISH = st.just("en")

_NO_ETYMOLOGY = st.just("")


def references(
    year: int,
) -> st.SearchStrategy[str]:
    """
    Draw a source naming one year, laid out as Wiktionary lays a date out.

    Args:
        year: The year it names, which is the earliest one it names.

    Returns:
        A strategy over the shapes such a reference takes.
    """
    return st.sampled_from(_REFERENCE_SHAPES).map(lambda shape: shape.format(year=year))


def _dotted(
    parts: tuple[str, str, int],
) -> str:
    """
    Join the parts of an identifier the way an extraction joins them.

    Args:
        parts: The headword, its part of speech and the ordinal telling it
            from the entries sharing both.

    Returns:
        The identifier, as bank.noun.2.
    """
    word, code, ordinal = parts

    return f"{word}.{code}.{ordinal}"


identifiers = st.tuples(
    words,
    pos_codes,
    st.integers(min_value=1, max_value=99),
).map(_dotted)
"""An identifier of the shape an extraction writes."""


@st.composite
def raw_examples(
    draw: st.DrawFn,
    texts: st.SearchStrategy[str] = texts,
    references: st.SearchStrategy[str | None] = _UNQUOTED,
    kinds: st.SearchStrategy[str | None] = _UNTYPED,
) -> RawJson:
    """
    Draw one sentence illustrating a sense, quoted from a source or not.

    Args:
        draw: Turns a strategy into one of its values.
        texts: The sentences to draw from.
        references: The sources to draw from, None writing no reference.
        kinds: What wiktextract read it as, None writing no kind at all.

    Returns:
        The example, as wiktextract writes one.
    """
    raw: RawJson = {"text": draw(texts)}

    for key, drawn in (("ref", draw(references)), ("type", draw(kinds))):
        if drawn is not None:
            raw[key] = drawn

    return raw


_EXAMPLES = st.lists(raw_examples(), max_size=2)


@st.composite
def raw_synonyms(
    draw: st.DrawFn,
    words: st.SearchStrategy[str] = words,
) -> RawJson:
    """
    Draw one word standing for the same meaning as a sense or an entry.

    Args:
        draw: Turns a strategy into one of its values.
        words: The words to draw the synonym from.

    Returns:
        The synonym, as wiktextract writes one.
    """
    return {"word": draw(words)}


_SYNONYMS = st.lists(raw_synonyms(), max_size=2)


@st.composite
def raw_senses(
    draw: st.DrawFn,
    glosses: st.SearchStrategy[list[str]] = _GLOSS_CHAINS,
    tags: st.SearchStrategy[list[str]] = _LABELS,
    topics: st.SearchStrategy[list[str]] = _LABELS,
    examples: st.SearchStrategy[list[RawJson]] = _EXAMPLES,
    synonyms: st.SearchStrategy[list[RawJson]] = _SYNONYMS,
) -> RawJson:
    """
    Draw one sense of an entry.

    A key holding nothing is left out rather than written empty, since
    wiktextract leaves it out and the two have to read alike.

    Args:
        draw: Turns a strategy into one of its values.
        glosses: The gloss chains to draw from, outermost first.
        tags: The labels of grammar and register to draw from.
        topics: The subject fields to draw from.
        examples: The sentences to draw from.
        synonyms: Other words for that meaning alone.

    Returns:
        The sense, as wiktextract writes one.
    """
    raw: RawJson = {"glosses": draw(glosses)}

    for key, drawn in (
        ("tags", draw(tags)),
        ("topics", draw(topics)),
        ("examples", draw(examples)),
        ("synonyms", draw(synonyms)),
    ):
        if drawn:
            raw[key] = drawn

    return raw


@st.composite
def raw_forms(
    draw: st.DrawFn,
    forms: st.SearchStrategy[str] = words,
    tags: st.SearchStrategy[list[str]] = _LABELS,
) -> RawJson:
    """
    Draw one written form of an entry, inflected or otherwise.

    Args:
        draw: Turns a strategy into one of its values.
        forms: The forms to draw from.
        tags: The labels wiktextract hangs off a form.

    Returns:
        The form, as wiktextract writes one.
    """
    return {"form": draw(forms), "tags": draw(tags)}


_SENSES = st.lists(raw_senses(), min_size=1, max_size=3)

_FORMS = st.lists(raw_forms(), max_size=2)


@st.composite
def raw_translations(
    draw: st.DrawFn,
    translations: st.SearchStrategy[str] = words,
    codes: st.SearchStrategy[str] = languages,
    glosses: st.SearchStrategy[str] = glosses,
) -> RawJson:
    """
    Draw one word another language uses for a sense of the entry.

    Args:
        draw: Turns a strategy into one of its values.
        translations: The words to draw the translation from.
        codes: The languages it may belong to.
        glosses: The meanings a translation table may head.

    Returns:
        The translation, as wiktextract writes one.
    """
    return {
        "word": draw(translations),
        "lang_code": draw(codes),
        "sense": draw(glosses),
    }


_TRANSLATIONS = st.lists(raw_translations(), max_size=2)


@st.composite
def raw_entries(
    draw: st.DrawFn,
    headwords: st.SearchStrategy[str] = words,
    pos_codes: st.SearchStrategy[str] = pos_codes,
    languages: st.SearchStrategy[str] = _ENGLISH,
    etymology_numbers: st.SearchStrategy[str] = _NO_ETYMOLOGY,
    etymology_texts: st.SearchStrategy[str] = _NO_ETYMOLOGY,
    forms: st.SearchStrategy[list[RawJson]] = _FORMS,
    senses: st.SearchStrategy[list[RawJson]] = _SENSES,
    translations: st.SearchStrategy[list[RawJson]] = _TRANSLATIONS,
) -> RawJson:
    """
    Draw one dictionary entry, which is what a lemma is read out of.

    Args:
        draw: Turns a strategy into one of its values.
        headwords: The words to draw from.
        pos_codes: Wiktextract's codes for a part of speech.
        languages: The languages a headword may belong to.
        etymology_numbers: Which etymology of the page it sits under.
        etymology_texts: What that etymology says.
        forms: The shapes a headword takes.
        senses: The meanings to hang off it.
        translations: What other languages call it.

    Returns:
        The entry, as wiktextract writes one.
    """
    raw: RawJson = {
        "word": draw(headwords),
        "pos": draw(pos_codes),
        "lang_code": draw(languages),
    }

    for key, drawn in (
        ("etymology_number", draw(etymology_numbers)),
        ("etymology_text", draw(etymology_texts)),
        ("forms", draw(forms)),
        ("senses", draw(senses)),
        ("translations", draw(translations)),
    ):
        if drawn:
            raw[key] = drawn

    return raw


# The models an export is handed, their text drawn wider than an extraction
# would hand over: quotation marks, newlines and the rest.

_LABEL_LISTS = st.lists(st.text(max_size=10), max_size=2).map(tuple)

sentences: st.SearchStrategy[Sentence] = st.one_of(
    st.builds(Example, texts, word_offsets=_WORD_OFFSETS),
    st.builds(
        Quotation,
        texts,
        st.text(min_size=1, max_size=20),
        st.none() | years,
        word_offsets=_WORD_OFFSETS,
    ),
)
"""One sentence a sense carries, of either kind."""

senses = st.builds(
    Sense,
    identifiers,
    st.lists(st.text(max_size=30), min_size=1, max_size=3).map(tuple),
    st.just("") | etymologies,
    _LABEL_LISTS,
    _LABEL_LISTS,
    _LABEL_LISTS,
    st.lists(sentences, max_size=3),
    _LABEL_LISTS,
)
"""One meaning of a lemma, filled the way an extraction fills it."""

translations = st.dictionaries(
    st.text(max_size=20),
    st.dictionaries(
        languages,
        st.lists(words, min_size=1, max_size=3).map(frozenset),
        max_size=2,
    ),
    max_size=2,
)
"""What other languages call a lemma, gathered under the glosses translated."""

lemmas = st.builds(
    Lemma,
    identifiers,
    words,
    parts_of_speech,
    st.lists(words, max_size=3).map(frozenset),
    st.lists(senses, max_size=3),
    translations,
)
"""One lemma, as an extraction hands it to a writer."""
