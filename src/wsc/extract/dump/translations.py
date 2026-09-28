"""
Translation tables read from raw Wiktionary markup.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import groupby

from ...constants.extraction import TRANSLATION_TEMPLATES
from ...identifiers import lemma_id, translation_table_id
from ...models import POS, TranslationTable
from ..translations import clean_translations, normalize_translation_gloss, templates
from .markup import plain, section_lines
from .source_markup import unwrap_mathematics

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


def _section_texts(
    markup: str,
    language_section: str,
) -> Iterator[tuple[POS, str]]:
    """
    Join consecutive section lines so templates may span line breaks.

    Args:
        markup: Page wikitext containing language and part-of-speech headings.
        language_section: Language heading whose sections should be read.

    Yields:
        Part of speech and complete text for each consecutive section.
    """
    sections = groupby(
        section_lines(markup, language_section), key=lambda item: item[0]
    )

    for pos, lines in sections:
        if pos is not None:
            yield pos, "\n".join(line for _, line in lines)


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

    for pos, section in _section_texts(markup, language_section):
        gloss = ""

        for name, parameters in templates(section):
            if name in {"trans-see", "trans-top-see"}:
                raw_heading, sources = unwrap_mathematics(
                    plain(parameters.get("1", ""))
                )

                heading = normalize_translation_gloss(
                    raw_heading,
                    mathematical_sources=sources,
                )

                targets = tuple(
                    plain(value)
                    for key, value in parameters.items()
                    if key.isdecimal() and int(key) > 1 and value
                )

                if heading:
                    pointers[pos][heading] = targets or (raw_heading,)

            elif name in {"trans-top", "trans-top-also"}:
                raw_heading, sources = unwrap_mathematics(parameters.get("1", ""))

                gloss = normalize_translation_gloss(
                    raw_heading,
                    mathematical_sources=sources,
                )

            elif name == "trans-bottom":
                gloss = ""

            elif gloss and name in TRANSLATION_TEMPLATES:
                translation = clean_translations(
                    parameters.get("1", ""),
                    parameters.get("2", ""),
                )

                if translation is not None:
                    language, words = translation

                    translated = tables[pos].setdefault(gloss, {})
                    translated[language] = translated.get(language, frozenset()) | words

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
