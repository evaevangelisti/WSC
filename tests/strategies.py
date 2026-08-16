"""
Generators the properties are drawn from.

A generator says what a source may write rather than what it usually writes:
a property is only worth stating if the entry nobody thought of can falsify
it, so the alphabets reach past ASCII and the lists reach down to empty.

A strategy is named for what it draws. One that has to be told something is a
function, so that a test spells out the part it rests on and leaves the rest
to be drawn.
"""

import string
from datetime import date

from hypothesis import strategies as st

from wsc.models import POS, Example, Lemma, Quotation, Sense, Sentence

type RawJson = dict[str, object]
"""One decoded JSON object, as wiktextract writes them."""

# Letters as far as Latin Extended-B. Wiktionary is written in more than
# these, but Python's re folds their case the way str does, and reading a
# form back out of a sentence rests on the two agreeing.
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

texts = st.text(min_size=1, max_size=60).filter(lambda text: bool(text.strip()))
"""The sentence an example or a quotation carries."""

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

dump_dates = st.dates(
    min_value=date(2001, 1, 1),
    max_value=date(2099, 12, 31),
).map(lambda day: day.strftime("%Y%m%d"))
"""The day a dump began, as the directory holding it is named."""

wordnet_versions = st.integers(min_value=1000, max_value=9999).map(str)
"""The year one edition of the wordnet came out."""

parts_of_speech: st.SearchStrategy[POS] = st.sampled_from(POS)
"""One part of speech the collector keeps."""

pos_codes = parts_of_speech.map(lambda pos: pos.value)
"""Wiktextract's code for one of them."""

unknown_pos_codes = st.text(
    alphabet=string.ascii_lowercase,
    min_size=1,
    max_size=6,
).filter(lambda code: code not in {pos.value for pos in POS})
"""A part of speech Wiktionary describes and the collector does not keep."""

# What a builder falls back on when a test has nothing to say about that part
# of an entry. A default draws what a source may hold rather than nothing at
# all, so that the part no test speaks for is still varied.

_OFFSETS = st.integers(min_value=0, max_value=60)

_WORD_OFFSETS = st.lists(st.tuples(_OFFSETS, _OFFSETS), max_size=3).map(tuple)

_UNQUOTED: st.SearchStrategy[str | None] = st.none()

_GLOSS_CHAINS = st.lists(glosses, min_size=1, max_size=3)

_LABELS = st.lists(words, max_size=2)

_ENGLISH = st.just("en")


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
) -> RawJson:
    """
    Draw one sentence illustrating a sense, quoted from a source or not.

    Args:
        draw: Turns a strategy into one of its values.
        texts: The sentences to draw from.
        references: The sources to draw from, None writing no reference.

    Returns:
        The example, as wiktextract writes one.
    """
    raw: RawJson = {"text": draw(texts)}

    reference = draw(references)
    if reference is not None:
        raw["ref"] = reference

    return raw


_EXAMPLES = st.lists(raw_examples(), max_size=2)


@st.composite
def raw_senses(
    draw: st.DrawFn,
    glosses: st.SearchStrategy[list[str]] = _GLOSS_CHAINS,
    tags: st.SearchStrategy[list[str]] = _LABELS,
    topics: st.SearchStrategy[list[str]] = _LABELS,
    examples: st.SearchStrategy[list[RawJson]] = _EXAMPLES,
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

    Returns:
        The sense, as wiktextract writes one.
    """
    raw: RawJson = {"glosses": draw(glosses)}

    for key, drawn in (
        ("tags", draw(tags)),
        ("topics", draw(topics)),
        ("examples", draw(examples)),
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
def raw_entries(
    draw: st.DrawFn,
    headwords: st.SearchStrategy[str] = words,
    pos_codes: st.SearchStrategy[str] = pos_codes,
    languages: st.SearchStrategy[str] = _ENGLISH,
    forms: st.SearchStrategy[list[RawJson]] = _FORMS,
    senses: st.SearchStrategy[list[RawJson]] = _SENSES,
) -> RawJson:
    """
    Draw one dictionary entry, which is what a lemma is read out of.

    Args:
        draw: Turns a strategy into one of its values.
        headwords: The words to draw from.
        pos_codes: Wiktextract's codes for a part of speech.
        languages: The languages a headword may belong to.
        forms: The shapes a headword takes.
        senses: The meanings to hang off it.

    Returns:
        The entry, as wiktextract writes one.
    """
    raw: RawJson = {
        "word": draw(headwords),
        "pos": draw(pos_codes),
        "lang_code": draw(languages),
    }

    for key, drawn in (("forms", draw(forms)), ("senses", draw(senses))):
        if drawn:
            raw[key] = drawn

    return raw


# The models an export is handed. Their text is drawn wider than an
# extraction would hand over, a writer having to survive whatever a source
# wrote: quotation marks, newlines and the rest.

_LABEL_LISTS = st.lists(st.text(max_size=10), max_size=2).map(tuple)

sentences: st.SearchStrategy[Sentence] = st.one_of(
    st.builds(Example, texts, word_offsets=_WORD_OFFSETS),
    st.builds(
        Quotation,
        texts,
        st.text(max_size=20),
        st.none() | years,
        word_offsets=_WORD_OFFSETS,
    ),
)
"""One sentence a sense carries, of either kind."""

senses = st.builds(
    Sense,
    identifiers,
    st.lists(st.text(max_size=30), min_size=1, max_size=3).map(tuple),
    _LABEL_LISTS,
    _LABEL_LISTS,
    st.lists(sentences, max_size=3),
    st.lists(st.text(max_size=16), max_size=2).map(tuple),
)
"""One meaning of a lemma, filled the way an extraction fills it."""

lemmas = st.builds(
    Lemma,
    identifiers,
    words,
    parts_of_speech,
    st.lists(senses, max_size=3),
)
"""One lemma, as an extraction hands it to a writer."""
