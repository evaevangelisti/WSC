"""The sentences illustrating one sense of an entry."""

import re

from ....models import (
    Attestation,
    Example,
    Quotation,
    Sentence,
    WordOffset,
    WordOffsetSource,
)
from ...markup import (
    BIBLIOGRAPHY,
    METADATA,
    NAVIGATION,
    is_literal_markup,
    is_unrecoverable,
    normalize_formatting,
    remove_references,
)
from ..schema import RawExample

_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")

_EXAMPLE = "example"
_QUOTATION = "quotation"


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
        return text, reference

    kind = raw_example.get("type", "")

    if kind == _EXAMPLE:
        return text, ""

    head, separator, tail = text.partition("\n")

    if not separator or not tail.strip():
        return text, ""

    if kind == _QUOTATION or parse_year(head) is not None:
        return tail.strip(), head.strip()

    return text, ""


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


def clean_sentence(
    value: Attestation,
    *,
    quoted: bool,
) -> Attestation | None:
    """
    Clean an attestation and relocate its existing word offsets.

    Args:
        value: Sentence text and ranges in its original coordinate system.
        quoted: Whether an actual source reference accompanies the sentence.

    Returns:
        A cleaned attestation, or None for metadata or unrecoverable fragments.
    """
    literal = is_literal_markup(value.text)
    value = normalize_formatting(value, preserve_markup=literal)

    if not value.text:
        return None

    if not literal and (
        is_unrecoverable(value.text) or "{{" in value.text or "}}" in value.text
    ):
        return None

    if not quoted and not literal:
        if METADATA.match(value.text) or NAVIGATION.match(value.text):
            return None

        if "\n" not in value.text and BIBLIOGRAPHY.match(value.text):
            return None

        value = remove_references(value, explicit=True)
        value = normalize_formatting(value, preserve_markup=True)

    return value if value.text else None


def parse_sentences(
    raw_examples: list[RawExample],
    minimum_year: int | None,
    maximum_year: int | None,
) -> list[Sentence]:
    """
    Collect the sentences illustrating one sense.

    The year filter reaches quotations alone: examples carry no reference, and so no
    date. Where the lemma falls is settled later.

    Args:
        raw_examples: What wiktextract listed under the sense.
        minimum_year: Oldest quotation to keep, or None for no bound.
        maximum_year: Newest quotation to keep, or None for no bound.

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
        )

        if cleaned is None:
            continue

        text, word_offsets = cleaned.text, cleaned.word_offsets

        if not reference:
            sentences.append(
                Example(
                    text,
                    word_offsets=word_offsets,
                )
            )

            continue

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
