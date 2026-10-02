"""
Remove bibliographic fragments embedded in unreferenced examples.
"""

import re

from .....models import Attestation
from ....markup import BIBLIOGRAPHY, normalize_formatting
from ....offsets import substitute
from ...schema import RawExample
from .references import EXAMPLE_KIND

_AUTHOR_CITATION = re.compile(
    r"^[A-Z][^\n]*?\b(?:18|19|20)\d{2}\.\s+[\"“][^\"”]+[\"”]\.\s+[^\n]+$"
)
_FIRST_LINE = re.compile(r"^[^\n]*\n")
_DOCUMENT_HEADER = re.compile(
    r"^[^\n]*\bvolume\s+\d+\s+no\.?\s+\d+\s+\(pdf\)\s+from[^\n]*\n",
    re.IGNORECASE,
)
_ABBREVIATED_SOURCE = re.compile(
    r"^(?:[A-Z]\.\s*)?[A-Z][a-z]+\.\s+(?:[A-Z][a-z]+\.\s*)+$"
)
_ANTDATED_SOURCE_HEADER = re.compile(r"^a(?:1[5-9]|20)\d{2}\s+[A-Z][^\n]*[\u02d0:]$")
_READ_AT_CITATION = re.compile(
    r"^a?(?:1[5-9]|20)\d{2}:[^\n]*?\bread at\b[^\n]*?\s+-\s+(?=[A-Z])",
    re.IGNORECASE,
)
_PUBLICATION_NOTE = re.compile(r"[ \t]+\[published in [^\[\]]+\]$", re.IGNORECASE)
_TRAILING_CITATION = re.compile(
    r"[ \t]+[—–-][ \t]+[^—–\n]*https?://\S+[ \t]*$", re.IGNORECASE
)
_PARENTHETICAL_CITATION = re.compile(
    r"[ \t]+\([^()\n]*https?://\S+[ \t]*$", re.IGNORECASE
)
_SOURCE_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BIBLIOGRAPHY_ONLY = re.compile(
    r"^(?:〃\s*,?\s*§\s*\d+(?:\.\d+)*,?\s*page\s+\d+|"
    + r"(?:18|19|20)\d{2},?\s+in\s+[A-Z].*|"
    + r"a(?:18|19|20)\d{2},\s+[A-Z][^,]+,\s+[^.]+\.)$",
    re.IGNORECASE,
)

_IDENTIFIER = re.compile(r"\b(?:(?i:ISBN|OCLC|ISSN|JSTOR)|DOI|doi(?=\s*:))\b")
_SOURCE_LINK = re.compile(
    r"(?:→|\\+to\s+)(?:ISBN|OCLC|ISSN|DOI|JSTOR)\b",
    re.IGNORECASE,
)
_CITATION_START = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"”]))[ \t]+(?=[A-Z0-9])",
)
_DATE = re.compile(r"\b(?:1[5-9]|20)\d{2}\b")
_AUTHOR = re.compile(r",\s+[A-Z]")
_HEADER = re.compile(
    r"^[,;]?\s*(?:(?:(?:c\.?|ca\.?|circa|early|late|mid|a\.|\(?transl\.\)?)\s*)*"
    + r"\d{1,2}(?:st|nd|rd|th)\s+century\b|"
    + r"(?:a|c\.\s*)?(?:(?:1[0-9]|20)\d{2}s?|\d{3}\?|20\?\?)"
    + r"(?:\s*[-–]\s*\d{2,4})?(?:\s*[,:'’\[(]|[A-Z]\.)|"
    + r"\d{1,2}C,|"
    + r"(?:[A-Z]\.\s*){1,3}[A-Z][a-z]+,|"
    + r"[A-Z][a-z]+\s+[A-Z][a-z]+,\s+[A-Z]|"
    + r"(?:Episode|Chapter|Volume|Book)\s+(?:\d+|[IVXLCDM]+)[,.:]|"
    + r"[^.!?\n]*\bSeason\s+\d+,\s+Episode\s+\d+\b)",
)
_PARENTHETICAL = re.compile(r"[ \t]+\((?:[^()]|\([^()]*\))*\)$")
_QUOTE_AT_START = re.compile(r"^[\"“](?P<sentence>.+?[.!?])[\"”](?=[ \t]+)")
_QUOTE_AT_END = re.compile(r"[\"“](?P<sentence>[^\"“”]+[.!?])[\"”][ \t]*$")


def is_standalone_bibliography(
    text: str,
) -> bool:
    """
    Identify a source citation that contains no usage example.

    Args:
        text: Unreferenced example text.

    Returns:
        Whether the text is only a bibliographic reference.
    """
    return "\n" not in text and bool(
        BIBLIOGRAPHY.match(text) or _BIBLIOGRAPHY_ONLY.fullmatch(text)
    )


def should_clean_bibliography(
    raw_example: RawExample,
    value: Attestation,
) -> bool:
    """
    Select examples with explicit source evidence or uncertain provenance.

    Args:
        raw_example: Wiktextract record and its example classification.
        value: Cleaned text before bibliographic removal.

    Returns:
        Whether bibliographic cleanup applies to this example.
    """
    if raw_example.get("type") != EXAMPLE_KIND or _IDENTIFIER.search(value.text):
        return True

    head, separator, _ = value.text.partition("\n")

    return bool(
        separator
        and (
            BIBLIOGRAPHY.match(head)
            or _HEADER.match(head)
            or _ABBREVIATED_SOURCE.fullmatch(head)
            or _ANTDATED_SOURCE_HEADER.fullmatch(head)
            or _READ_AT_CITATION.match(value.text)
            or _DOCUMENT_HEADER.match(value.text)
        )
    )


def _retain_sentence(
    value: Attestation,
    start: int,
    end: int,
) -> Attestation:
    """
    Retain a quoted sentence and relocate its existing word offsets.

    Args:
        value: Example containing a sentence and source details.
        start: First character of the sentence to retain.
        end: Boundary after the sentence to retain.

    Returns:
        The quoted sentence with its original offset provenance.
    """
    suffix = value.text[end:]

    if suffix:
        value = substitute(value, re.compile(re.escape(suffix) + r"$"), "")

    prefix = value.text[:start]

    if prefix:
        value = substitute(value, re.compile(r"^" + re.escape(prefix)), "")

    return value


def _remove_quoted_citation(
    value: Attestation,
) -> Attestation:
    """
    Keep a quoted sentence when a dated citation encloses it.

    Args:
        value: Example containing a bibliographic identifier.

    Returns:
        The sentence alone, or the original text if no quote is isolated.
    """
    for pattern in (_QUOTE_AT_START, _QUOTE_AT_END):
        quotation = pattern.search(value.text)

        if quotation is None:
            continue

        start, end = quotation.span("sentence")
        remainder = (
            value.text[end:] if pattern is _QUOTE_AT_START else value.text[:start]
        )

        if _IDENTIFIER.search(remainder) and _DATE.search(remainder):
            return _retain_sentence(value, start, end)

    return value


def _remove_identifier_citation(
    value: Attestation,
) -> Attestation | None:
    """
    Remove citations identified by a publication identifier and date.

    Args:
        value: Example containing an ISBN, DOI, or similar identifier.

    Returns:
        Recoverable usage text, or None when its boundary is unknown.
    """
    value = _remove_quoted_citation(value)

    parenthetical = _PARENTHETICAL.search(value.text)

    if (
        parenthetical is not None
        and value.text[: parenthetical.start()].rstrip().endswith((".", "!", "?"))
        and _IDENTIFIER.search(parenthetical[0])
        and _DATE.search(parenthetical[0])
    ):
        value = substitute(value, _PARENTHETICAL, "")

    for boundary in _CITATION_START.finditer(value.text):
        tail = value.text[boundary.start() :]
        first_clause = tail.lstrip().split(".", 1)[0]

        if (
            (_DATE.search(first_clause) or _AUTHOR.search(first_clause))
            and _DATE.search(tail)
            and _IDENTIFIER.search(tail)
        ):
            value = substitute(value, re.compile(re.escape(tail) + r"$"), "")
            break

    if (
        _SOURCE_LINK.search(value.text)
        or _HEADER.match(value.text)
        or _IDENTIFIER.match(value.text)
    ):
        return None

    return value


def remove_example_bibliography(
    value: Attestation,
) -> Attestation | None:
    """
    Remove unstructured source details without losing example offsets.

    Args:
        value: Unreferenced example and its known word offsets.

    Returns:
        The example without source details, or None if no example remains.
    """
    if _AUTHOR_CITATION.match(value.text) and _SOURCE_URL.search(value.text):
        return None

    if _IDENTIFIER.search(value.text):
        cleaned = _remove_identifier_citation(value)

        if cleaned is None:
            return None

        value = cleaned

    head, separator, _ = value.text.partition("\n")

    if separator and (
        BIBLIOGRAPHY.match(head)
        or _HEADER.match(head)
        or _ABBREVIATED_SOURCE.fullmatch(head)
        or _ANTDATED_SOURCE_HEADER.fullmatch(head)
        or _DOCUMENT_HEADER.match(value.text)
    ):
        value = substitute(value, _FIRST_LINE, "")

    value = substitute(value, _READ_AT_CITATION, "")
    value = substitute(value, _PUBLICATION_NOTE, "")
    value = substitute(value, _TRAILING_CITATION, "")
    value = substitute(value, _PARENTHETICAL_CITATION, "")
    value = substitute(value, _SOURCE_URL, "")

    value = normalize_formatting(value, preserve_markup=True)

    return value if any(character.isalnum() for character in value.text) else None
