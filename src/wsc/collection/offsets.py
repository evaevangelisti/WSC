"""Inspect offset coverage and agreement between extraction methods."""

import re
import unicodedata
from collections import Counter

from ..models import WordOffset, WordOffsetSource

_MARKS = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-"})

_WORD = re.compile(r"\w")


def classify_headword(
    lemma: str,
) -> str:
    """
    Say what a headword looks like, where no occurrence of it was found.

    The first class that fits wins, so a headword is counted once. A plain one is the
    interesting case: nothing about it explains the miss.

    Args:
        lemma: The headword.

    Returns:
        The class it falls in.
    """
    if " " in lemma:
        return "Multiword"

    if "'" in lemma or "’" in lemma:
        return "Apostrophe"

    if "-" in lemma:
        return "Hyphenated"

    if not lemma.isascii():
        return "Non-ASCII"

    return "Plain"


def contains_headword(
    lemma: str,
    text: str,
) -> bool:
    """
    Say whether a sentence spells a headword the extractor did not find.

    Case and typographic punctuation are unified first. Word boundaries are kept, so a
    headword inside a longer word still does not count.

    Args:
        lemma: The headword.
        text: The sentence.

    Returns:
        Whether it is in there after all.
    """

    def normalise(
        value: str,
    ) -> str:
        """
        Unify punctuation, Unicode forms, and letter case.

        Args:
            value: Headword or sentence to normalize.

        Returns:
            Text suitable for normalized literal matching.
        """
        return unicodedata.normalize("NFKC", value.translate(_MARKS)).casefold()

    return (
        re.search(rf"(?<!\w){re.escape(normalise(lemma))}(?!\w)", normalise(text))
        is not None
    )


def check_offsets(
    text: str,
    offsets: tuple[WordOffset, ...],
) -> Counter[str]:
    """
    Hold one sentence's offsets against what the extractor promises.

    These checks identify violations of the extractor's offset contract.

    Args:
        text: The sentence the offsets index.
        offsets: The offsets, as the export writes them.

    Returns:
        Violations counted separately for each supporting source.
    """
    broken: Counter[str] = Counter()
    previous_ends = dict.fromkeys(WordOffsetSource, 0)

    for word_offset in offsets:
        start, end = word_offset.offset

        for source in word_offset.sources:
            if not 0 <= start < end <= len(text):
                broken[f"Out of range ({source})"] += 1
                continue

            if start < previous_ends[source]:
                broken[f"Overlapping or unordered ({source})"] += 1

            previous_ends[source] = end

            if text[start:end] != text[start:end].strip():
                broken[f"Padded with space ({source})"] += 1

            before = text[start - 1] if start else ""
            after = text[end] if end < len(text) else ""

            if _WORD.match(before) or _WORD.match(after):
                broken[f"Inside a longer word ({source})"] += 1

    return broken


def compare_sources(
    offsets: tuple[WordOffset, ...],
) -> str:
    """
    Describe how both methods located one sentence.

    Args:
        offsets: Candidate ranges and their supporting methods.

    Returns:
        The relationship between bold and lemmatizer proposals.
    """
    bold = {
        tuple(word_offset.offset)
        for word_offset in offsets
        if "bold" in word_offset.sources
    }

    lemmatizer = {
        tuple(word_offset.offset)
        for word_offset in offsets
        if "lemmatizer" in word_offset.sources
    }

    if not bold:
        return "Lemmatizer only"

    if not lemmatizer:
        return "Bold only"

    if bold == lemmatizer:
        return "Exact agreement"

    if bold & lemmatizer:
        return "Partial agreement"

    if any(
        bold_start < lemmatizer_end and lemmatizer_start < bold_end
        for bold_start, bold_end in bold
        for lemmatizer_start, lemmatizer_end in lemmatizer
    ):
        return "Overlapping proposals"

    return "Disagreement"
