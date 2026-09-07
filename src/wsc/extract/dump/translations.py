"""The translation tables Wiktionary writes away from the entry they belong to."""

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from ...models import POS, Translations
from .markup import arguments, plain

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

_POS_BY_HEADING: dict[str, POS] = {
    "Noun": POS.NOUN,
    "Proper noun": POS.NAME,
    "Verb": POS.VERB,
    "Adjective": POS.ADJECTIVE,
    "Adverb": POS.ADVERB,
}

_TOP = re.compile(r"\{\{trans-top(?:-also)?\s*\|([^{}]*)\}\}")
_BOTTOM = re.compile(r"\{\{trans-bottom\s*\}\}")

_TRANSLATION = re.compile(
    r"\{\{(?:t|t\+|tt|tt\+|t-check|t\+check|t-simple)\|([^{}]*)\}\}"
)

_SEE = re.compile(r"\{\{trans-(?:see|top-see)\s*\|([^{}]*)\}\}")

SUBPAGE_SUFFIX = "/translations"


@dataclass(frozen=True, slots=True)
class PageTranslations:
    """
    What one page translates for one part of speech, away from the entry.

    Attributes:
        lemma: The headword the translations belong to.
        pos: Its part of speech.
        translations: What a subpage of the entry holds, by gloss.
        pointers: Which headwords translate a gloss, by gloss.
    """

    lemma: str
    pos: POS
    translations: Translations = field(default_factory=dict)
    pointers: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _read_pointers(
    line: str,
) -> Iterator[tuple[str, tuple[str, ...]]]:
    """
    Read every pointer one line writes at the headwords translating a meaning.

    Args:
        line: The line to read.

    Yields:
        The meaning, and the headwords said to translate it.
    """
    for found_pointer in _SEE.finditer(line):
        positional_arguments, _ = arguments(found_pointer.group(1))

        if not positional_arguments:
            continue

        gloss = plain(positional_arguments[0])

        if not gloss:
            continue

        # Single-argument templates use the same value for meaning and target headword.

        pointed_lemmas = tuple(plain(name) for name in positional_arguments[1:])

        yield gloss, pointed_lemmas or (gloss,)


def _read_translations(
    line: str,
) -> Iterator[tuple[str, str]]:
    """
    Read every translation one line of a table offers.

    Args:
        line: The line to read.

    Yields:
        The language offering a word, by code, and the word.
    """
    for found_translation in _TRANSLATION.finditer(line):
        positional_arguments, _ = arguments(found_translation.group(1))

        if len(positional_arguments) < 2:
            continue

        language, word = positional_arguments[0], plain(positional_arguments[1])

        if language and word:
            yield language, word


def _sections(
    markup: str,
    language_section: str,
) -> Iterator[tuple[POS, str]]:
    """
    Walk the lines of a page sitting under a part of speech we keep.

    Args:
        markup: What the page is written in.
        language_section: What the edition heads its own language with.

    Yields:
        The part of speech a line sits under, and the line.
    """
    reading_language = False
    pos: POS | None = None

    for line in markup.splitlines():
        found_heading = _HEADING.match(line)

        if not found_heading:
            if reading_language and pos is not None:
                yield pos, line

            continue

        heading = found_heading.group(2)

        if len(found_heading.group(1)) == 2:
            reading_language = heading == language_section
            pos = None

        found_pos = _POS_BY_HEADING.get(heading)

        if found_pos is not None:
            pos = found_pos


def read_page(
    title: str,
    markup: str,
    language_section: str,
) -> Iterator[PageTranslations]:
    """
    Read the translations one page keeps away from the entry.

    A subpage holds the tables, an entry the pointers.

    Args:
        title: The page, which names the headword.
        markup: What the page is written in.
        language_section: What the edition heads its own language with.

    Yields:
        One record per part of speech the page translates.
    """
    subpage = title.endswith(SUBPAGE_SUFFIX)

    tables: defaultdict[POS, Translations] = defaultdict(dict)
    pointers: defaultdict[POS, dict[str, tuple[str, ...]]] = defaultdict(dict)

    previous_pos: POS | None = None
    gloss: str | None = None

    for pos, line in _sections(markup, language_section):
        if pos is not previous_pos:
            previous_pos, gloss = pos, None

        pointers[pos].update(_read_pointers(line))

        found_top = _TOP.search(line)

        if found_top:
            positional_arguments, _ = arguments(found_top.group(1))
            headed = plain(positional_arguments[0]) if positional_arguments else ""

            gloss = headed or None

        if _BOTTOM.search(line):
            gloss = None

        if not subpage or gloss is None:
            continue

        translated_words = tables[pos].setdefault(gloss, {})

        for language, word in _read_translations(line):
            translated_words[language] = translated_words.get(language, frozenset()) | {
                word
            }

    for part_of_speech in sorted(tables.keys() | pointers.keys()):
        yield PageTranslations(
            title.removesuffix(SUBPAGE_SUFFIX),
            part_of_speech,
            {
                gloss: translated_words
                for gloss, translated_words in tables.get(part_of_speech, {}).items()
                if translated_words
            },
            dict(pointers.get(part_of_speech, {})),
        )
