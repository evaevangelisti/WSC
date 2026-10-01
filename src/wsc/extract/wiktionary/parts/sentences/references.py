"""
Read and normalize references attached to Wiktionary examples.
"""

import re

from ....markup import normalize_statement
from ...schema import RawExample

EXAMPLE_KIND = "example"
_QUOTATION = "quotation"
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")
_INLINE_QUOTATION = re.compile(
    r"^(?P<reference>.+\b(?:18|19|20)\d{2}\b.+):[ \t]*[“\"]" + r"(?P<text>.+)$",
    re.DOTALL,
)


def parse_year(
    reference: str,
) -> int | None:
    """
    Read the year of publication off a reference.

    Args:
        reference: The source, as Wiktionary formats it.

    Returns:
        The first year the reference names, or None if it names none.
    """
    found_year = _YEAR_PATTERN.search(reference)

    return int(found_year.group(1)) if found_year else None


def clean_reference(
    reference: str,
) -> str:
    """
    Normalize surrounding whitespace and stray reference delimiters.

    Args:
        reference: The source reference as Wiktionary formats it.

    Returns:
        The reference without trailing colons or an unmatched opening bracket.
    """
    reference = reference.strip()

    if reference.startswith("[") and reference.count("[") > reference.count("]"):
        reference = reference[1:].lstrip()

    return normalize_statement(reference)


def read_source(
    text: str,
    raw_example: RawExample,
) -> tuple[str, str]:
    """
    Tell a sentence apart from the source it was taken from.

    Embedded quotation references are separated from the sentence text.

    Args:
        text: The sentence, as wiktextract wrote it.
        raw_example: What it listed beside it.

    Returns:
        The sentence, and the source naming it, empty where there is none.
    """
    reference = raw_example.get("ref", "").strip()

    if reference:
        return text, clean_reference(reference)

    kind = raw_example.get("type", "")

    if kind == EXAMPLE_KIND:
        return text, ""

    if inline_quotation := _INLINE_QUOTATION.fullmatch(text):
        return inline_quotation["text"].rstrip('”"').rstrip(), clean_reference(
            inline_quotation["reference"]
        )

    head, separator, tail = text.partition("\n")

    if not separator or not tail.strip():
        return text, ""

    if kind == _QUOTATION or parse_year(head) is not None:
        return tail.strip(), clean_reference(head)

    return text, ""
