"""What other languages call an entry."""

from collections import defaultdict

from ....models import TranslationTable
from ...identifiers import translation_table_id
from ..merge import add_translations
from ..schema import RawTranslation

# Placeholder headings require full matches to preserve meaningful glosses.
_PLACEHOLDER_GLOSSES = frozenset(
    {
        "translations",
        "translations to be checked",
        "translations to be specified",
        "translation gloss",
        "sense",
    }
)


def parse_translations(
    raw_translations: list[RawTranslation],
    off_page_translations: tuple[TranslationTable, ...] | None = None,
    lemma_id: str = "",
) -> tuple[TranslationTable, ...]:
    """
    Gather an entry's translations under the glosses heading them.

    Wiktionary stores translation tables at entry level.

    Args:
        raw_translations: What wiktextract listed under the entry.
        off_page_translations: Optional translations from linked pages.
        lemma_id: The entry identifier used to name tables.

    Returns:
        The words each language offers for each gloss translated.
    """
    gathered_translations: defaultdict[str, defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for raw_translation in raw_translations:
        gloss = raw_translation.get("sense", "").strip()
        language = raw_translation.get("lang_code", "").strip()
        word = raw_translation.get("word", "").strip()

        if not gloss or not language or not word:
            continue

        if gloss.casefold() in _PLACEHOLDER_GLOSSES:
            continue

        gathered_translations[gloss][language].add(word)

    translations = tuple(
        TranslationTable(
            translation_table_id(lemma_id, gloss),
            gloss,
            {language: frozenset(words) for language, words in translated.items()},
        )
        for gloss, translated in gathered_translations.items()
    )

    if off_page_translations:
        translations = add_translations(
            translations,
            off_page_translations,
            lemma_id,
        )

    return translations
