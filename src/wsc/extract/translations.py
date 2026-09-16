"""Normalization shared by translation sources."""

import re

_SEE_ALSO = re.compile(
    r"\s+[—–-]\s+see also\b.*$",
    re.IGNORECASE | re.DOTALL,
)


def normalize_translation_gloss(
    gloss: str,
) -> str:
    """
    Normalize a translation gloss for grouping and comparison.

    Wiktextract appends displayed ``see also`` references to some table glosses.

    Args:
        gloss: Gloss read from Wiktextract or a translation template.

    Returns:
        Stripped gloss without a trailing ``see also`` reference.
    """
    return _SEE_ALSO.sub("", gloss).strip()


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
