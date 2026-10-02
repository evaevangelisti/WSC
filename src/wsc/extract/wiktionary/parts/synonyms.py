"""
Other words standing for what a sense means.
"""

import re

from ..schema import RawSynonym

_BROKEN_MARKUP = re.compile(r"\]\]>|\[\[|\]\]|\{\{|\}\}")
_THESAURUS_ANNOTATION = re.compile(r"\s*\[⇒\s*thesaurus\]", re.IGNORECASE)
_REGISTER_ANNOTATION = re.compile(r"\s+\((?:slang)\)$", re.IGNORECASE)


def _has_balanced_delimiters(
    word: str,
) -> bool:
    """
    Check that parenthetical and square annotations are complete.

    Args:
        word: A candidate synonym from Wiktextract.

    Returns:
        Whether its delimiters close in the right order.
    """
    openings: list[str] = []

    for character in word:
        if character in "([":
            openings.append(character)
        elif character in ")]" and (
            not openings or openings.pop() != ("(" if character == ")" else "[")
        ):
            return False

    return not openings


def parse_synonyms(
    raw_synonyms: list[RawSynonym],
    lemma: str,
) -> tuple[str, ...]:
    """
    Read the synonyms wiktextract tied to one sense.

    Args:
        raw_synonyms: What wiktextract listed under the sense.
        lemma: The headword, which is no synonym of itself.

    Returns:
        The words offered for that meaning alone, a repeat kept once.
    """
    seen: dict[str, None] = {}

    for raw_synonym in raw_synonyms:
        word = raw_synonym.get("word", "").strip()
        word = _THESAURUS_ANNOTATION.sub("", word)
        word = _REGISTER_ANNOTATION.sub("", word).strip()

        if (
            any(character.isalnum() for character in word)
            and word != lemma
            and not word.endswith((":", ",", ";"))
            and not _BROKEN_MARKUP.search(word)
            and _has_balanced_delimiters(word)
        ):
            seen[word] = None

    return tuple(seen)
