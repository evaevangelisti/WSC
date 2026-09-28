"""
What other languages call an entry.
"""

from collections import defaultdict
from itertools import chain

from ....identifiers import translation_table_id
from ....models import Attestation, TranslationTable
from ...dump.source_markup import MathSource, restore_mathematics
from ...translations import clean_translations, normalize_translation_gloss
from ..schema import RawTranslation


def parse_translations(
    raw_translations: list[RawTranslation],
    off_page_translations: tuple[TranslationTable, ...] | None = None,
    lemma_id: str = "",
    *,
    mathematics: tuple[MathSource, ...] = (),
) -> tuple[TranslationTable, ...]:
    """
    Gather clean translations from both sources under their meaning headings.

    Args:
        raw_translations: What Wiktextract listed under the entry.
        off_page_translations: Optional translations from linked pages.
        lemma_id: The entry identifier used to name tables.
        mathematics: Original formulae and their source contexts.

    Returns:
        The words each language offers for each usable definition.
    """
    gathered_translation_tables: defaultdict[str, defaultdict[str, set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )

    source_translations = (
        (
            restore_mathematics(Attestation(item.get("sense", "")), mathematics).text,
            item.get("lang_code", ""),
            item.get("word", ""),
        )
        for item in raw_translations
    )

    supplementary_translations = (
        (table.gloss, language, word)
        for table in off_page_translations or ()
        for language, words in table.translations.items()
        for word in words
    )

    for heading, code, written in chain(
        source_translations,
        supplementary_translations,
    ):
        gloss = normalize_translation_gloss(
            heading,
            mathematical_sources=tuple(formula.source for formula in mathematics),
        )

        translation = clean_translations(code, written)

        if gloss and translation is not None:
            language, words = translation
            gathered_translation_tables[gloss][language].update(words)

    return tuple(
        TranslationTable(
            translation_table_id(lemma_id, gloss),
            gloss,
            {language: frozenset(words) for language, words in translated.items()},
        )
        for gloss, translated in gathered_translation_tables.items()
    )
