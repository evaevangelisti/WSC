"""
Remove navigation and editorial references from definitions.
"""

import re

from ...models import Attestation
from ..offsets import substitute
from .formatting import (
    is_literal_markup,
    is_unrecoverable,
    normalize_formatting,
    normalize_statement,
)

__all__ = [
    "BIBLIOGRAPHY",
    "METADATA",
    "NAVIGATION",
    "clean_definition_references",
    "is_literal_markup",
    "is_unrecoverable",
    "normalize_formatting",
    "normalize_statement",
    "remove_references",
]

METADATA = re.compile(
    r"^(?:near[- ]synonyms?|meronyms?|holonyms?|usage notes?):", re.IGNORECASE
)

BIBLIOGRAPHY = re.compile(
    r"^(?:c\.|ca\.|circa)?\s*[12]\d{3}(?:\s*[-–—]\s*\d{2,4})?"
    + r"(?:\s+[A-Z][a-z]+\.?(?:\s+\d{1,2})?)?\s*[,:]\s*[\"“'‘(]?[A-Z]"
)

NAVIGATION = re.compile(
    r"^(?:[*•#-]\s*)?(?:[.;]\s*(?:compare|cf\.)\s+|"
    + r"(?:but(?:\s+also)?|also)\s+see(?:\s*:\s*|\s+)|see\s*:\s*|"
    + r"see\s+(?:also\b|(?:Citations|Thesaurus|Appendix|Wikipedia):|"
    + r"(?:the\s+)?(?:quotations?|usage notes?|translations?)\b)|"
    + r"for\s+(?:examples?|quotations?)\b[^\n]*\bsee\b(?=\s+\S))",
    re.IGNORECASE,
)

_SEE = (
    r"(?:(?:but(?:\s+also)?|also)\s+)?"
    + r"see(?:\s*:\s*|,\s*for example,\s*|\s+(?:also\s+)?)"
)
_PURPOSE = r"(?:for\s+[^();\n]*?,\s*|for\s+[^();\n]*?\bforms\s+|more formally,\s*)"
_DIRECTIVE = rf"(?:{_PURPOSE})?(?:{_SEE}|cf\.\s+|compare(?:\s+with)?\s+)"
_BODY = r"(?:[^.\n()]|\.(?!\s|$)|\((?:[^()]|\([^()]*\))*\))+"

_START = re.compile(rf"^(?:[*•#-]\s*)?{_SEE}", re.IGNORECASE)
_PURPOSE_START = re.compile(rf"^{_PURPOSE}{_SEE}", re.IGNORECASE)
_DEFINITION = re.compile(r"^see\s+[^\n]+?,\s*for:\s*", re.IGNORECASE)

_PARENTHESES = re.compile(
    rf"[ \t]*\(\s*{_DIRECTIVE}[^()]*(?:\([^()]*\)[^()]*)*\)", re.IGNORECASE
)

_UNCLOSED_REFERENCE = re.compile(
    rf"[ \t]*\(\s*{_DIRECTIVE}[^()]*(?:\([^()]*\)[^()]*)*$", re.IGNORECASE
)

_CLAUSE = re.compile(
    rf"(?P<before>[.;:,]|[ \t]*[—–-])\s*{_DIRECTIVE}"
    + _BODY
    + r"(?:\.(?=\s|$)|(?=\)|$))",
    re.IGNORECASE,
)

_LEXICAL_OBJECT = re.compile(
    rf"^,\s*{_SEE}(?:it|them|him|her|us|me|you|what|how|whether|if|that)\b",
    re.IGNORECASE,
)

_UNPUNCTUATED_COMPARISON = re.compile(r"[ \t]+cf\.\s+" + _BODY + r"\.?$")

_QUOTED_TARGET = re.compile(
    r"(?<=[\"”'])\s+see:\s*" + _BODY + r"(?=\)|$)", re.IGNORECASE
)

_TABLE_TAIL = re.compile(
    r"\s+[—–-]\s+(?:otherwise\s+see|see\s+also)\b.*$", re.IGNORECASE | re.DOTALL
)

_SOURCE = re.compile(
    r"\s*\((?:Source:\s*)?https?://[^\s()]+\)|\s*\(Source:[^()]+https?://[^()]+\)",
    re.IGNORECASE,
)

_URL = re.compile(r"[ \t]*https?://[^\s<>]+(?:[ \t]+https?://[^\s<>]+)*")

_FURTHER_REFERENCE = re.compile(
    r"[ \t]*For (?:more|details|examples)\b[^()\n]*?\bsee\s+https?://[^\s()]+",
    re.IGNORECASE,
)

_DEMONSTRATION = re.compile(
    r"\s*For video demonstration, click here:\s*https?://\S+", re.IGNORECASE
)

_EXPLICIT = re.compile(
    r"\b(?:compare\s+|(?:but(?:\s+also)?|also)\s+see\s*:|"
    + r"see\s*:\s*|see\s+(?:also\b|"
    + r"(?:Citations|Thesaurus|Wikipedia|Appendix):|usage notes?\b))",
    re.IGNORECASE,
)

_EMPTY = re.compile(r"\([ \t]*\)|[ \t]+([,.;:])")


def remove_references(
    value: Attestation,
    *,
    explicit: bool = False,
) -> Attestation:
    """
    Remove bounded references, requiring explicit editorial cues in examples.

    Args:
        value: Text and known word offsets before reference removal.
        explicit: Whether each removed reference must contain an editorial cue.

    Returns:
        Text with bounded references removed and surviving offsets relocated.
    """
    original = value

    for pattern in (_PARENTHESES, _UNCLOSED_REFERENCE):
        value = substitute(
            value,
            pattern,
            lambda match: (
                match[0] if explicit and not _EXPLICIT.search(match[0]) else ""
            ),
        )

    value = substitute(
        value,
        _CLAUSE,
        lambda match: (
            match[0]
            if (explicit and not _EXPLICIT.search(match[0]))
            or _LEXICAL_OBJECT.match(match[0])
            else "."
            if match["before"] == "."
            else ""
        ),
    )

    if value.text != original.text:
        value = substitute(value, _EMPTY, lambda match: match[1] or "")

    return value


def clean_definition_references(
    text: str,
    *,
    table: bool = False,
) -> str:
    """
    Keep definitions while removing navigation and editorial source URLs.

    Args:
        text: A definition with possible navigation or source references.
        table: Whether the definition heads a translation table.

    Returns:
        The retained definition, or an empty string for standalone navigation.
    """
    if text.casefold().startswith(("see;", "see.")):
        return text

    value = substitute(Attestation(text), _DEFINITION, "")

    if (
        NAVIGATION.match(value.text)
        or _PURPOSE_START.match(value.text)
        or (not table and _START.match(value.text))
    ):
        return ""

    if table:
        value = substitute(value, _TABLE_TAIL, "")

    value = remove_references(value, explicit=text.casefold().startswith("to see "))
    value = substitute(value, _UNPUNCTUATED_COMPARISON, "")
    value = substitute(value, _QUOTED_TARGET, "")

    value = substitute(value, _DEMONSTRATION, "")
    value = substitute(value, _FURTHER_REFERENCE, "")
    value = substitute(value, _SOURCE, "")

    value = substitute(
        value,
        _URL,
        lambda match: (
            "."
            if match[0].endswith(".")
            and not match.string[: match.start()].rstrip().endswith((".", "!", "?"))
            else ""
        ),
    )

    return value.text.strip()
