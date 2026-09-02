"""
The sentences illustrating one sense of an entry.
"""

import re

from ...models import Example, Quotation, Sentence
from ..schema import RawExample

# A year of its own, from the first century of printing up to this one, or
# the decade it opens.
_YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})s?\b")

# What wiktextract calls each kind of sentence, where it names one.
_EXAMPLE = "example"
_QUOTATION = "quotation"

# What the seeCites template leaves in place of a sentence, pointing at a
# page of quotations rather than attesting anything.
_POINTER_PATTERN = re.compile(r"^For quotations using this term, see Citations:")


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

    Where wiktextract hands no source over, a quotation carries it on the
    first line of its own text, which is split off rather than searched.

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


def parse_sentences(
    raw_examples: list[RawExample],
    minimum_year: int | None,
    maximum_year: int | None,
) -> list[Sentence]:
    """
    Collect the sentences illustrating one sense.

    The year filter reaches quotations alone: examples carry no reference,
    and so no date. Where the lemma falls is settled later.

    Args:
        raw_examples: What wiktextract listed under the sense.
        minimum_year: Oldest quotation to keep, or None for no bound.
        maximum_year: Newest quotation to keep, or None for no bound.

    Returns:
        The sentences that survive it, in the order they were listed.
    """
    sentences: list[Sentence] = []

    for raw_example in raw_examples:
        text = raw_example.get("text", "").strip()
        if not text:
            continue

        text, reference = read_source(text, raw_example)

        if not reference:
            # A pointer is only ever left where no kind was read.
            if "type" not in raw_example and _POINTER_PATTERN.match(text):
                continue

            sentences.append(Example(text))

            continue

        year = parse_year(reference)

        # An undated quotation only stands in the way once a bound is set.
        if minimum_year is not None or maximum_year is not None:
            if year is None:
                continue

            if minimum_year is not None and year < minimum_year:
                continue

            if maximum_year is not None and year > maximum_year:
                continue

        sentences.append(Quotation(text, reference, year))

    return sentences
