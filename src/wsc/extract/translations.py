"""Read translation templates and normalize tables consistently across sources."""

import re
from collections.abc import Iterator

from ..constants.extraction import (
    SEA_LANGUAGES,
    TRANSLATION_PLACEHOLDERS,
    TRANSLATION_TEMPLATES,
)
from ..models import Attestation
from .markup import (
    BIBLIOGRAPHY,
    NAVIGATION,
    clean_definition_references,
    is_literal_markup,
    is_unrecoverable,
    normalize_formatting,
    remove_references,
)

_OPENINGS = {"{{{": "}}}", "{{": "}}", "[[": "]]"}

_OPAQUE = re.compile(
    r"<!--.*?-->|<nowiki\b[^>]*>.*?</nowiki>", re.IGNORECASE | re.DOTALL
)

_NAMED = re.compile(r"^([A-Za-z0-9_-]+)=(.*)$", re.DOTALL)


def _delimiters(
    text: str,
) -> Iterator[tuple[int, int, str]]:
    """
    Locate outer templates and argument separators outside nested constructs.

    Args:
        text: Wikitext whose nested constructs delimit template arguments.

    Yields:
        Start, end, and kind of each outer template or separator.
    """
    stack: list[str] = []

    start = 0
    position = 0

    while position < len(text):
        opaque = _OPAQUE.match(text, position)

        if opaque is not None:
            position = opaque.end()
            continue

        if stack and text.startswith(stack[-1], position):
            position += len(stack.pop())

            if not stack and text.startswith("{{", start):
                yield start, position, "template"

            continue

        opening = next(
            (token for token in _OPENINGS if text.startswith(token, position)), None
        )

        if opening is not None:
            if not stack:
                start = position

            stack.append(_OPENINGS[opening])
            position += len(opening)

            continue

        if text[position] == "|" and not stack:
            yield position, position + 1, "separator"

        position += 1


def _arguments(
    body: str,
) -> dict[str, str]:
    """
    Read parameters in source order without splitting nested constructs.

    Args:
        body: Template arguments following the template name.

    Returns:
        Named and positional arguments, with later assignments taking precedence.
    """
    parameters: dict[str, str] = {}

    boundaries = [start for start, _, kind in _delimiters(body) if kind == "separator"]

    position = 0
    number = 1

    for boundary in (*boundaries, len(body)):
        argument = body[position:boundary].strip()
        found = _NAMED.match(argument)

        if found is None:
            parameters[str(number)] = argument
            number += 1
        else:
            parameters[found[1]] = found[2].strip()

        position = boundary + 1

    return parameters


def templates(
    text: str,
) -> Iterator[tuple[str, dict[str, str]]]:
    """
    Read complete outer templates in document order.

    Args:
        text: Wikitext containing templates and surrounding prose.

    Yields:
        Template names and parameters keyed by name or decimal position.
    """
    for start, end, kind in _delimiters(text):
        if kind != "template" or text.startswith("{{{", start):
            continue

        body = text[start + 2 : end - 2]
        name, separator, body = body.partition("|")

        yield name.strip(), _arguments(body) if separator else {}


def read_template(
    text: str,
) -> tuple[str, dict[str, str]] | None:
    """
    Read a single complete template without ignoring surrounding fragments.

    Args:
        text: Wikitext expected to contain exactly one complete template.

    Returns:
        The template name and arguments, or None for incomplete or extra content.
    """
    boundaries = list(_delimiters(text))

    if boundaries != [(0, len(text), "template")]:
        return None

    return next(templates(text), None)


_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)*")
_START = re.compile(r"^see\s+", re.IGNORECASE)
_ENTRY = re.compile(r"^see entry\)\s*", re.IGNORECASE)
_INVISIBLE = re.compile("[\\s\u00ad\u200b-\u200f\u2060\ufeff]*")


def clean_translation(
    language: str,
    word: str,
) -> tuple[str, str] | None:
    """
    Recover complete translation markup and exclude reference instructions.

    Args:
        language: The source's language code, including dialect subtags.
        word: A translated word or recognized translation template.

    Returns:
        The cleaned language and word, or None when no translation is recoverable.
    """
    language, word = language.strip(), word.strip()

    if word.startswith("{{") and word.endswith("}}"):
        parsed = read_template(word)

        if parsed is None or parsed[0] not in TRANSLATION_TEMPLATES:
            return None

        parameters = parsed[1]
        language, word = (
            parameters.get("1", "").strip(),
            parameters.get("2", "").strip(),
        )

    if _LANGUAGE.fullmatch(language) is None:
        return None

    word = normalize_formatting(Attestation(word), preserve_joiners=True).text
    word = _ENTRY.sub("", word)

    if is_unrecoverable(word) or any(
        marker in word for marker in ("{{", "}}", "[[", "]]")
    ):
        return None

    if NAVIGATION.match(word) or word.casefold().startswith("etc. see "):
        return None

    if _START.match(word) and not (
        language in SEA_LANGUAGES and word.startswith("See ")
    ):
        return None

    word = remove_references(Attestation(word)).text.strip()

    return (language, word) if word and _INVISIBLE.fullmatch(word) is None else None


def normalize_translation_gloss(
    gloss: str,
) -> str:
    """
    Normalize a translation gloss for grouping and comparison.

    All sources share the same formatting, navigation, and placeholder rules.

    Args:
        gloss: Gloss read from Wiktextract or a translation template.

    Returns:
        A normalized definition, or an empty string for an unusable heading.
    """
    literal = is_literal_markup(gloss)
    text = normalize_formatting(Attestation(gloss), preserve_markup=literal).text

    if not literal:
        text = clean_definition_references(text, table=True)

    if not literal and (is_unrecoverable(text) or "{{" in text or "}}" in text):
        return ""

    if "[[" in text or "]]" in text or BIBLIOGRAPHY.match(text):
        return ""

    text = normalize_formatting(Attestation(text), preserve_markup=True).text

    return "" if " ".join(text.casefold().split()) in TRANSLATION_PLACEHOLDERS else text


def translation_gloss_key(
    gloss: str,
) -> str:
    """
    Build a presentation-independent key for comparing translation glosses.

    Args:
        gloss: Gloss whose capitalization, spacing, and final period may vary.

    Returns:
        Case-folded words separated by single spaces, without one final period.
    """
    key = " ".join(normalize_translation_gloss(gloss).split()).casefold()

    return key[:-1] if key.endswith(".") and not key.endswith("..") else key
