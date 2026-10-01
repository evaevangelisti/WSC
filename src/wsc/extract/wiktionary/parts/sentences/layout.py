"""
Split layout-separated examples without losing word offsets.
"""

import re

from .....models import Attestation, WordOffset
from ....offsets import substitute

EXAMPLE_SEPARATOR = re.compile(r"[ \t]*\u2003+[ \t]*")
_EXAMPLE_ENDINGS = frozenset(";.?!")


def split_examples(
    value: Attestation,
) -> tuple[Attestation, ...]:
    """
    Split layout-separated examples only when their boundaries are supported.

    Args:
        value: Cleaned unreferenced example and its known word offsets.

    Returns:
        Separate examples, or one example with normalized layout spacing.
    """
    separators = tuple(EXAMPLE_SEPARATOR.finditer(value.text))

    if not separators:
        return (value,)

    boundaries: list[tuple[int, int]] = []
    start = 0

    for separator in separators:
        boundaries.append((start, separator.start()))
        start = separator.end()

    boundaries.append((start, len(value.text)))

    all_segments_have_offsets = all(
        any(
            left <= offset.offset[0] < offset.offset[1] <= right
            for offset in value.word_offsets
        )
        for left, right in boundaries
    )

    all_boundaries_are_punctuated = all(
        value.text[left:right].rstrip().endswith(tuple(_EXAMPLE_ENDINGS))
        for left, right in boundaries[:-1]
    )

    has_repeated_separator = any(match[0].count("\u2003") > 1 for match in separators)

    if not (
        all_segments_have_offsets
        or all_boundaries_are_punctuated
        or has_repeated_separator
    ):
        return (substitute(value, EXAMPLE_SEPARATOR, " "),)

    examples: list[Attestation] = []

    for left, right in boundaries:
        segment = value.text[left:right]
        leading_space = len(segment) - len(segment.lstrip())
        text = segment.strip()

        if text.endswith(";"):
            text = text[:-1].rstrip()

        segment_start = left + leading_space
        word_offsets = tuple(
            WordOffset(
                (
                    offset.offset[0] - segment_start,
                    offset.offset[1] - segment_start,
                ),
                offset.sources,
            )
            for offset in value.word_offsets
            if segment_start
            <= offset.offset[0]
            < offset.offset[1]
            <= segment_start + len(text)
        )

        if text:
            examples.append(Attestation(text, word_offsets=word_offsets))

    return tuple(examples)
