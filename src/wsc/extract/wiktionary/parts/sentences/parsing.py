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
_SPACED_SENTENCE = re.compile(r"(?<=[.!?])[ \t]{2,}(?=\S)")

_SCORE_LINE = re.compile(r"\n\s*\{")
_TRAILING_SCORE_NUMBER = re.compile(r"\n\s*\d+\s*$")
_LILYPOND_SCORE = re.compile(
    r"\{[^{}]*(?:\\(?:key|clef|time|relative|set)\b|[a-g](?:is|es)?')[^{}]*\}",
    re.DOTALL,
)
_SCORE_TEMPLATE = re.compile(r"^\{\{(?:ux|uxi|usex)\|en\|", re.IGNORECASE)
_AUDIO_PLACEHOLDER = re.compile(
    r"audio playback is not supported in your browser", re.IGNORECASE
)
_NUMERIC_ONLY = re.compile(r"\d+(?:[\s.,:;/-]\d+)*")
_EDITORIAL_EXAMPLE = re.compile(
    r"^(?:(?:alternative forms?|coordinate terms?|related terms?|synonyms?|antonyms?)"
    + r"(?:\s+\([^)]*\))?:|\(initialisms?\)$)",
    re.IGNORECASE,
)
_BIBLIOGRAPHY_ONLY = re.compile(
    r"^(?:〃\s*,?\s*§\s*\d+(?:\.\d+)*,?\s*page\s+\d+|"
    + r"(?:18|19|20)\d{2},?\s+in\s+[A-Z].*)$",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_IDENTIFIER = re.compile(
    r"(?:→|\\+to\s+)(?:ISBN|OCLC|ISSN|DOI|JSTOR)\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_START = re.compile(
    r"(?<=[.!?])[ \t]+(?=(?:(?:18|19|20)\d{2}\b|"
    + r"[A-Z][\w’-]*(?:\s+[A-Z][\w’-]*){1,3},\s+[A-Z]))",
)


def _preserve_example_boundaries(
    value: Attestation,
) -> Attestation:
    """
    Keep explicit layout spacing through formatting normalization.

    Args:
        value: Unquoted example with its source offsets.

    Returns:
        Example whose potential separators remain distinguishable.
    """
    value = substitute(value, _SPACED_SEMICOLON, ";\u2003")

    if len(value.word_offsets) > 1:
        value = substitute(value, _SPACED_SENTENCE, "\u2003")

    return value


def _has_media(
    text: str,
    score_prefixes: tuple[str, ...] | None,
) -> bool:
    """
    Identify unquoted examples whose content is an audio or score rendering.

    Args:
        text: Example text emitted by Wiktextract.
        score_prefixes: Source text before score tags on the same page.

    Returns:
        Whether the example contains media rather than lexical usage.
    """
    if (
        _SCORE_LINE.search(text)
        or _LILYPOND_SCORE.search(text)
        or _AUDIO_PLACEHOLDER.search(text)
    ):
        return True

    for prefix in score_prefixes or ():
        rendered_prefix = re.split(
            r"<|[^\x00-\x7f]",
            _SCORE_TEMPLATE.sub("", prefix),
            maxsplit=1,
        )[0].strip()

        if (
            len(rendered_prefix) >= 12
            and text.startswith(rendered_prefix)
            and _TRAILING_SCORE_NUMBER.search(text)
        ):
            return True

    return False


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

    if stripped_text.endswith(text):
        return len(stripped_text) - len(text)

    if text in stripped_text:
        return stripped_text.rfind(text)

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
) -> Attestation | None:
    """
    Remove metadata from an unquoted example.

    Args:
        value: Example after formatting and reference lines are removed.

    Returns:
        The remaining example, or None when it contains only metadata.
    """
    value = normalize_formatting(value, preserve_markup=True)

    if (
        METADATA.match(value.text)
        or NAVIGATION.match(value.text)
        or _EDITORIAL_EXAMPLE.match(value.text)
    ):
        return None

    if "\n" not in value.text and (
        BIBLIOGRAPHY.match(value.text) or _BIBLIOGRAPHY_ONLY.fullmatch(value.text)
    ):
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

    if not quoted and _has_media(value.text, score_prefixes):
        return None

    value = restore_mathematics(value, mathematics)

    if not quoted and not literal:
        value = _preserve_example_boundaries(value)

    value = normalize_formatting(value, preserve_markup=literal)

    if not literal:
        value = substitute(value, _REFERENCE_LINE, "")
        value = substitute(value, _TITLE_REFERENCE, "")

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
        cleaned = _clean_unquoted_sentence(value)

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

    if _BIBLIOGRAPHIC_IDENTIFIER.search(value.text):
        for boundary in _BIBLIOGRAPHIC_START.finditer(value.text):
            tail = value.text[boundary.start() :]

            if parse_year(tail) is not None and _BIBLIOGRAPHIC_IDENTIFIER.search(tail):
                value = substitute(value, re.compile(re.escape(tail) + r"$"), "")
                break
        else:
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
