"""Translation tables read from raw Wiktionary markup."""

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field

from ...identifiers import lemma_id, translation_table_id
from ...models import POS, TranslationTable
from ..translations import normalize_translation_gloss
from .markup import arguments, plain

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

_POS_BY_HEADING: dict[str, POS] = {
    "Noun": POS.NOUN,
    "Proper noun": POS.PROPN,
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
    What one page translates for one part of speech.

    Attributes:
        lemma: The headword the translations belong to.
        pos: Its part of speech.
        translations: What a subpage of the entry holds.
        pointers: Which headwords translate a gloss, by gloss.
    """

    lemma: str
    pos: POS
    translations: tuple[TranslationTable, ...] = ()
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

        gloss = normalize_translation_gloss(plain(positional_arguments[0]))

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
) -> Iterator[tuple[POS | None, str]]:
    """
    Walk the lines of a page sitting under a part of speech we keep.

    Args:
        markup: What the page is written in.
        language_section: What the edition heads its own language with.

    Yields:
        The current part of speech and line, or None at a section boundary.
    """
    reading_language = False

    pos: POS | None = None
    pos_level = 0

    for line in markup.splitlines():
        found_heading = _HEADING.match(line)

        if not found_heading:
            if reading_language and pos is not None:
                yield pos, line

            continue

        heading = found_heading.group(2)
        level = len(found_heading.group(1))

        if level == 2:
            reading_language = heading == language_section

        if level <= pos_level:
            pos = None

        found_pos = _POS_BY_HEADING.get(heading)

        if reading_language and found_pos is not None:
            pos = found_pos
            pos_level = level

        yield None, line


def read_page(
    title: str,
    markup: str,
    language_section: str,
) -> Iterator[PageTranslations]:
    """
    Read the translations and pointers written on one page.

    Main pages and translation subpages can both hold tables.

    Args:
        title: The page, which names the headword.
        markup: What the page is written in.
        language_section: What the edition heads its own language with.

    Yields:
        One record per part of speech the page translates.
    """
    tables: defaultdict[POS, dict[str, dict[str, frozenset[str]]]] = defaultdict(dict)
    pointers: defaultdict[POS, dict[str, tuple[str, ...]]] = defaultdict(dict)

    gloss: str | None = None

    for pos, line in _sections(markup, language_section):
        if pos is None:
            gloss = None

            continue

        pointers[pos].update(_read_pointers(line))

        found_top = _TOP.search(line)

        if found_top:
            positional_arguments, _ = arguments(found_top.group(1))
            headed = (
                normalize_translation_gloss(plain(positional_arguments[0]))
                if positional_arguments
                else ""
            )

            gloss = headed or None

        if _BOTTOM.search(line):
            gloss = None

        if gloss is None:
            continue

        translations = tables[pos].setdefault(gloss, {})

        for language, word in _read_translations(line):
            translations[language] = translations.get(language, frozenset()) | {word}

    for part_of_speech in sorted(tables.keys() | pointers.keys()):
        yield PageTranslations(
            title.removesuffix(SUBPAGE_SUFFIX),
            part_of_speech,
            tuple(
                TranslationTable(
                    translation_table_id(
                        lemma_id(title.removesuffix(SUBPAGE_SUFFIX), part_of_speech),
                        gloss,
                    ),
                    gloss,
                    translations,
                )
                for gloss, translations in tables.get(part_of_speech, {}).items()
                if translations
            ),
            dict(pointers.get(part_of_speech, {})),
        )
