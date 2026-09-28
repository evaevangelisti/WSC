"""
The sentences illustrating one sense of an entry.
"""

import re

from .....models import (
    Attestation,
    Example,
    Quotation,
    Sentence,
    WordOffset,
    WordOffsetSource,
)
from ....dump.source_markup import MathSource, restore_mathematics
from ....markup import (
    BIBLIOGRAPHY,
    METADATA,
    NAVIGATION,
    is_literal_markup,
    is_unrecoverable,
    normalize_formatting,
    remove_references,
)
from ....offsets import substitute
from ...schema import RawExample
from .layout import EXAMPLE_SEPARATOR, split_examples
from .references import EXAMPLE_KIND, clean_reference, parse_year, read_source

__all__ = [
    "clean_reference",
    "clean_sentence",
    "parse_sentences",
    "parse_year",
    "read_source",
]


_REFERENCE_LINE = re.compile(
    r"(?:^|\n)[ \t]*(?:[*•#-][ \t]+)?(?:(?:also|but)\s+)?"
    + r"see\s*(?::\s*)?(?:also\s*:?\s+)?"
    + r"(?:quotations?\s+(?:under|at)\b|(?:Citations|Thesaurus|Appendix|Wikipedia):)"
    + r"[^\n]*(?=\n|$)",
    re.IGNORECASE,
)

_TITLE_REFERENCE = re.compile(
    r"[ \t]*(?:\(\s*|\[\s*)(?:(?:also|but(?:\s+also)?)\s+)?"
    + r"see\s+(?:the\s+)?title(?:\s+already)?\.?\s*(?:\)|\])[ \t]*",
    re.IGNORECASE,
)

_TITLE_ONLY = re.compile(
    r"^(?:(?:also|but(?:\s+also)?)\s+)?see\s+(?:the\s+)?title\.?$",
    re.IGNORECASE,
)

_AUTHOR_CITATION = re.compile(
    r"^[A-Z][^\n]*?\b(?:18|19|20)\d{2}\.\s+[\"“][^\"”]+[\"”]\.\s+[^\n]+$"
)
_FIRST_LINE = re.compile(r"^[^\n]*\n")
_DOCUMENT_HEADER = re.compile(
    r"^[^\n]*\bvolume\s+\d+\s+no\.?\s+\d+\s+\(pdf\)\s+from[^\n]*\n",
    re.IGNORECASE,
)
_TRAILING_CITATION = re.compile(
    r"[ \t]+[—–-][ \t]+[^—–\n]*https?://\S+[ \t]*$", re.IGNORECASE
)
_PARENTHETICAL_CITATION = re.compile(
    r"[ \t]+\([^()\n]*https?://\S+[ \t]*$", re.IGNORECASE
)
_SOURCE_URL = re.compile(r"https?://\S+", re.IGNORECASE)

_SPACED_SEMICOLON = re.compile(r";[ \t]{2,}")

_MUSIC_SCORE = re.compile(r"\{\\(?:key|clef|time)\b[^{}]*\}")
_TRAILING_SCORE_NUMBER = re.compile(r"\n\s*\d+\s*$")
_NUMERIC_ONLY = re.compile(r"\d+(?:[\s.,:;/-]\d+)*")


def _sentence_start(
    raw_text: str,
    text: str,
) -> int:
    """
    Return the sentence's start within Wiktextract text.

    Args:
        raw_text: Complete text supplied by Wiktextract.
        text: Exported sentence without its reference.

    Returns:
        The sentence's code-point position within the stripped source text.
    """
    stripped_text = raw_text.strip()

    if stripped_text == text:
        return 0

    head, separator, tail = stripped_text.partition("\n")

    return len(head) + len(separator) + len(tail) - len(tail.lstrip())


def _parse_bold_offsets(
    raw_text: str,
    text: str,
    raw_offsets: list[list[int]],
) -> tuple[WordOffset, ...]:
    """
    Read valid bold ranges relative to the exported sentence.

    Args:
        raw_text: Complete text supplied by Wiktextract.
        text: Exported sentence without its reference.
        raw_offsets: Bold ranges relative to the complete text.

    Returns:
        Distinct valid ranges relative to the exported sentence.
    """
    leading_space = len(raw_text) - len(raw_text.lstrip())
    sentence_start = _sentence_start(raw_text, text)

    offset_shift = leading_space + sentence_start

    return tuple(
        dict.fromkeys(
            WordOffset(
                (start - offset_shift, end - offset_shift),
                (WordOffsetSource.BOLD,),
            )
            for start, end in raw_offsets
            if 0 <= start - offset_shift < end - offset_shift <= len(text)
        )
    )


def _clean_unquoted_sentence(
    value: Attestation,
    score_prefixes: tuple[str, ...] | None = None,
) -> Attestation | None:
    """
    Remove metadata and audio-score fragments from an unquoted example.

    Args:
        value: Example after formatting and reference lines are removed.
        score_prefixes: Source contexts for examples containing scores.

    Returns:
        The remaining example, or None when it contains only metadata.
    """
    if score_prefixes is None or any(
        value.text.startswith(prefix) for prefix in score_prefixes
    ):
        value = substitute(value, _TRAILING_SCORE_NUMBER, "")
    value = normalize_formatting(value, preserve_markup=True)

    if METADATA.match(value.text) or NAVIGATION.match(value.text):
        return None

    if "\n" not in value.text and BIBLIOGRAPHY.match(value.text):
        return None

    value = remove_references(value, explicit=True)

    return normalize_formatting(value, preserve_markup=True)


def clean_sentence(
    value: Attestation,
    *,
    quoted: bool,
    mathematics: tuple[MathSource, ...] = (),
    score_prefixes: tuple[str, ...] | None = None,
) -> Attestation | None:
    """
    Clean an attestation and relocate its existing word offsets.

    Args:
        value: Sentence text and ranges in its original coordinate system.
        quoted: Whether an actual source reference accompanies the sentence.
        mathematics: Original formulae and their source contexts.
        score_prefixes: Source contexts for examples containing scores.

    Returns:
        A cleaned attestation, or None for metadata or unrecoverable fragments.
    """
    literal = is_literal_markup(value.text)

    value = restore_mathematics(value, mathematics)

    if not quoted and not literal:
        value = substitute(value, _SPACED_SEMICOLON, ";\u2003")

    value = normalize_formatting(value, preserve_markup=literal)

    if not literal:
        value = substitute(value, _REFERENCE_LINE, "")
        value = substitute(value, _TITLE_REFERENCE, "")

        if not quoted:
            value = substitute(value, _MUSIC_SCORE, "")

        value = normalize_formatting(value, preserve_markup=True)

    if not value.text:
        return None

    if not literal and _TITLE_ONLY.fullmatch(value.text):
        return None

    if not literal and is_unrecoverable(
        value.text,
        mathematical_sources=tuple(formula.source for formula in mathematics),
    ):
        return None

    if not quoted and not literal:
        cleaned = _clean_unquoted_sentence(value, score_prefixes)

        if cleaned is None:
            return None

        value = cleaned

    if not quoted and _NUMERIC_ONLY.fullmatch(value.text):
        return None

    return value if any(character.isalnum() for character in value.text) else None


def _remove_example_bibliography(
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

    head, separator, _ = value.text.partition("\n")

    if separator and (BIBLIOGRAPHY.match(head) or _DOCUMENT_HEADER.match(value.text)):
        value = substitute(value, _FIRST_LINE, "")

    value = substitute(value, _TRAILING_CITATION, "")
    value = substitute(value, _PARENTHETICAL_CITATION, "")
    value = substitute(value, _SOURCE_URL, "")

    value = normalize_formatting(value, preserve_markup=True)

    return value if any(character.isalnum() for character in value.text) else None


def parse_sentences(
    raw_examples: list[RawExample],
    minimum_year: int | None,
    maximum_year: int | None,
    *,
    mathematics: tuple[MathSource, ...] = (),
    score_prefixes: tuple[str, ...] | None = None,
) -> list[Sentence]:
    """
    Collect the sentences illustrating one sense.

    The year filter reaches quotations alone: examples carry no reference, and so no
    date. Where the lemma falls is settled later.

    Args:
        raw_examples: What wiktextract listed under the sense.
        minimum_year: Oldest quotation to keep, or None for no bound.
        maximum_year: Newest quotation to keep, or None for no bound.
        mathematics: Original formulae and their source contexts.
        score_prefixes: Source contexts for examples containing scores.

    Returns:
        The sentences that survive it, in the order they were listed.
    """
    sentences: list[Sentence] = []

    for raw_example in raw_examples:
        raw_text = raw_example.get("text", "")

        text = raw_text.strip()

        if not text:
            continue

        text, reference = read_source(text, raw_example)

        word_offsets = _parse_bold_offsets(
            raw_text,
            text,
            raw_example.get("bold_text_offsets", []),
        )

        cleaned = clean_sentence(
            Attestation(text, word_offsets=word_offsets),
            quoted=bool(reference),
            mathematics=mathematics,
            score_prefixes=score_prefixes,
        )

        if (
            cleaned is not None
            and not reference
            and raw_example.get("type") != EXAMPLE_KIND
        ):
            cleaned = _remove_example_bibliography(cleaned)

        if cleaned is None:
            continue

        text, word_offsets = cleaned.text, cleaned.word_offsets

        if not reference:
            sentences.extend(
                Example(
                    example.text,
                    word_offsets=example.word_offsets,
                )
                for example in split_examples(
                    Attestation(text, word_offsets=word_offsets),
                )
            )

            continue

        cleaned = substitute(cleaned, EXAMPLE_SEPARATOR, " ")
        text, word_offsets = cleaned.text, cleaned.word_offsets

        year = parse_year(reference)

        if minimum_year is not None or maximum_year is not None:
            if year is None:
                continue

            if minimum_year is not None and year < minimum_year:
                continue

            if maximum_year is not None and year > maximum_year:
                continue

        sentences.append(
            Quotation(
                text,
                reference,
                year=year,
                word_offsets=word_offsets,
            )
        )

    return sentences
