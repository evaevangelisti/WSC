"""Read translation templates and normalize tables consistently across sources."""

import re
from collections.abc import Iterator
from itertools import product

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
    normalize_statement,
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
_START = re.compile(
    r"^(?:(?:but(?:\s+also)?|also)\s+see(?:\s*:\s*|\s+|$)|" + r"see(?:\s*:\s*|\s+))",
    re.IGNORECASE,
)
_ENTRY = re.compile(r"^see entry\)\s*", re.IGNORECASE)
_INVISIBLE = re.compile("[\\s\u00ad\u200b-\u200f\u2060\ufeff]*")

_MISSING_TRANSLATION = re.compile(
    r"\[\s*(?:script needed|translit\??|…|\.\.\.)\s*\]",
    re.IGNORECASE,
)
_TERM_PLACEHOLDER = re.compile(r"\[Term\?]", re.IGNORECASE)

_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
_NOUN_CLASS = re.compile(
    r"\s+class\s+[^\s/]+(?:/[^\s]+)?(?:\s+animate)?$",
    re.IGNORECASE,
)
_SQUARE_ANNOTATION = re.compile(r"(?<!\S)\[[^\[\]]+\](?!\S)")
_LEADING_SQUARE_ANNOTATION = re.compile(r"^\[[^\[\]]+\](?=\S)")
_UNFINISHED_SQUARE_ANNOTATION = re.compile(r"\s+\[[^\[\]]*$")

_GRAMMATICAL_SUFFIX = re.compile(
    r"\s+(?:(?:[cfmn]\s+)?(?:sg|pl)(?:\s+(?:and|or))?|"
    + r"[cfmn]\s+(?:singular|plural|and|or)|(?:un)?countable|impf|pf)\.?$",
    re.IGNORECASE,
)
_GENDER_SUFFIX = re.compile(r"\s+[cfmn](?:/[cfmn])*$", re.IGNORECASE)
_GENDER_BEFORE_SLASH = re.compile(r"\s+[cfmn](?=\s*[/\\]|$)", re.IGNORECASE)
_LEADING_GENDER = re.compile(r"^[cfmn]\s+", re.IGNORECASE)

_GENDER_ALTERNATIVE = re.compile(
    r"\s+[cfmn](?:/[cfmn])?\s+or\s+",
    re.IGNORECASE,
)
_MIDDLE_GENDER = re.compile(
    r"\s+([mf])(?:\s+(?:sg|pl))?(?:\s+(?:and|or))?(?=\s+\S)",
    re.IGNORECASE,
)
_SQUARE_OPTIONAL = re.compile(r"\[([^\[\]]+)\]")
_SPACED_SLASH = re.compile(r"\s+[/\\]\s*|\s*[/\\]\s+")
_SLASH_OR_BACKSLASH = re.compile(r"[/\\]")

_SPACE = re.compile(r"\s+")


def _remove_parentheticals(
    word: str,
) -> str:
    """
    Remove parenthetical annotations, including nested groups.

    Args:
        word: Translation with possible parenthetical annotations.

    Returns:
        Translation without complete parenthetical annotations.
    """
    previous = ""

    while word != previous:
        previous = word
        word = _PARENTHETICAL.sub("", word)

    return word


def _expand_square_optionals(
    word: str,
) -> tuple[str, ...]:
    """
    Expand square-bracketed text attached directly to lexical material.

    Args:
        word: Translation after standalone annotations were removed.

    Returns:
        Forms both without and with each optional fragment.
    """
    forms = (word,)

    while any(_SQUARE_OPTIONAL.search(form) for form in forms):
        expanded: list[str] = []

        for form in forms:
            found = _SQUARE_OPTIONAL.search(form)

            if found is None:
                expanded.append(form)
                continue

            included_prefix = form[: found.start()]

            if found[1].casefold() == "a" and included_prefix.endswith("o"):
                included_prefix = included_prefix[:-1]

            expanded.extend(
                (
                    f"{form[: found.start()]}{form[found.end() :]}",
                    f"{included_prefix}{found[1]}{form[found.end() :]}",
                ),
            )

        forms = tuple(expanded)

    return forms


def _split_gender_alternatives(
    word: str,
) -> tuple[str, ...]:
    """Split gender-marked forms when their boundaries are supported.

    Args:
        word: Translation that may embed masculine or feminine labels.

    Returns:
        Lexical segments, or none when a label leaves their roles uncertain.
    """
    markers = tuple(_MIDDLE_GENDER.finditer(word))

    if not markers:
        return (word,)

    segments = tuple(segment.strip() for segment in _MIDDLE_GENDER.split(word)[::2])

    if not all(segments):
        return ()

    if len(markers) > 1:
        return segments

    left_words, right_words = segments[0].split(), segments[1].split()

    if len(left_words) == len(right_words) == 1:
        return (
            segments
            if left_words[0][:3].casefold() == right_words[0][:3].casefold()
            else ()
        )

    if len(left_words) > 1 and len(right_words) > 1:
        return segments if left_words[0].casefold() == right_words[0].casefold() else ()

    return ()


def _expand_alternatives(
    word: str,
) -> tuple[str, ...]:
    """Expand explicit and optional alternatives with their shared context.

    Args:
        word: Translation containing alternative forms or segments.

    Returns:
        Complete lexical forms derived from the translation.
    """
    complete_alternatives = tuple(
        _LEADING_GENDER.sub("", alternative)
        for gender_alternative in _GENDER_ALTERNATIVE.split(word)
        for gender_segment in _split_gender_alternatives(gender_alternative)
        for alternative in _SPACED_SLASH.split(gender_segment)
    )

    return tuple(
        " ".join(words).strip()
        for alternative in complete_alternatives
        if alternative
        for optional in _expand_square_optionals(alternative)
        for words in product(
            *(_SLASH_OR_BACKSLASH.split(token) for token in optional.split())
        )
        if all(words)
    )


def clean_translations(
    language: str,
    word: str,
) -> tuple[str, frozenset[str]] | None:
    """
    Recover lexical translations and exclude editorial annotations.

    Args:
        language: The source's language code, including dialect subtags.
        word: A translated word or recognized translation template.

    Returns:
        The language and lexical alternatives, or None when none are recoverable.
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

    if _MISSING_TRANSLATION.fullmatch(word):
        return None

    if bracketed_word := _SQUARE_OPTIONAL.fullmatch(word):
        word = bracketed_word[1].strip()

    word = remove_references(Attestation(word)).text.strip()
    word = _remove_parentheticals(word)
    word = _SQUARE_ANNOTATION.sub("", word)
    word = _LEADING_SQUARE_ANNOTATION.sub("", word)
    word = _UNFINISHED_SQUARE_ANNOTATION.sub("", word)
    word = _TERM_PLACEHOLDER.sub("", word)
    word = _NOUN_CLASS.sub("", word)
    word = _GRAMMATICAL_SUFFIX.sub("", word)
    word = _GENDER_SUFFIX.sub("", word)
    word = _GENDER_BEFORE_SLASH.sub(" ", word)
    word = _LEADING_GENDER.sub("", word)
    word = _SPACE.sub(" ", word).strip(" /,;:")

    if "(" in word or ")" in word:
        return None

    alternatives = frozenset(
        cleaned
        for alternative in _expand_alternatives(word)
        if (cleaned := alternative.strip(" /,;:"))
        and _INVISIBLE.fullmatch(cleaned) is None
        and "[" not in cleaned
        and "]" not in cleaned
    )

    return (language, alternatives) if alternatives else None


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
    text = normalize_statement(text)

    placeholder = " ".join(text.casefold().split()).rstrip(".?!…‽")

    return "" if placeholder in TRANSLATION_PLACEHOLDERS else text


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
