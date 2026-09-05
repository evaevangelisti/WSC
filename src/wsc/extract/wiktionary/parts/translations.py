"""
What other languages call an entry.
"""

from collections import defaultdict

from ....models import Translations
from ..merge import add_translations
from ..schema import RawTranslation

# What a translation table is headed with where nobody wrote a gloss. Matched
# whole rather than by prefix: "translation studies" is a meaning, and so is
# "translation of source code into object code by a compiler".
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
    off_page_translations: Translations | None = None,
) -> Translations:
    """
    Gather an entry's translations under the glosses heading them.

    Wiktionary hangs a table off the entry, not off a sense.

    Args:
        raw_translations: What wiktextract listed under the entry.
        off_page_translations: What the dump holds away from the entry, or
        None where the dump was not walked for it.

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

    translations: Translations = {
        gloss: {language: frozenset(words) for language, words in translated.items()}
        for gloss, translated in gathered_translations.items()
    }

    if off_page_translations:
        add_translations(translations, off_page_translations)

    return translations
