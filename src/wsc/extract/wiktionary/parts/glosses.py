"""Classify glosses that describe forms, variants, and synonyms."""

import re

from ....constants.glosses import (
    FORM_GLOSS_PREFIXES,
    SYNONYM_GLOSS_PREFIXES,
    VARIANT_GLOSS_PREFIXES,
)
from ....models import Attestation
from ...markup import (
    BIBLIOGRAPHY,
    clean_definition_references,
    is_literal_markup,
    is_unrecoverable,
    normalize_definition_punctuation,
    normalize_formatting,
)

_TARGET_SEPARATOR = re.compile(r"\s*(?:[;,]|\(|$)")


def _matches_prefix(
    gloss: str,
    prefixes: frozenset[str],
) -> bool:
    """
    Match normalized prefixes at word boundaries.

    Args:
        gloss: Definition whose initial words identify its category.
        prefixes: Normalized category prefixes.

    Returns:
        Whether a complete prefix begins the normalized gloss.
    """
    normalized = " ".join(gloss.casefold().split())

    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in prefixes
    )


def is_form_gloss(
    gloss: str,
) -> bool:
    """
    Say whether a gloss describes an inflected form.

    Args:
        gloss: The gloss to inspect.

    Returns:
        Whether the gloss describes an inflected form.
    """
    return _matches_prefix(gloss, FORM_GLOSS_PREFIXES)


def is_synonym_gloss(
    gloss: str,
) -> bool:
    """
    Say whether a gloss describes a synonym.

    Args:
        gloss: The gloss to inspect.

    Returns:
        Whether the gloss describes a synonym.
    """
    return _matches_prefix(gloss, SYNONYM_GLOSS_PREFIXES)


def is_variant_gloss(
    gloss: str,
) -> bool:
    """
    Say whether a gloss describes a non-inflected variant.

    Args:
        gloss: The gloss to inspect.

    Returns:
        Whether the gloss describes a non-inflected variant.
    """
    return _matches_prefix(gloss, VARIANT_GLOSS_PREFIXES)


def referenced_lemma(
    gloss: str,
) -> str | None:
    """
    Extract the first referenced lemma from a variant gloss.

    Args:
        gloss: The gloss containing a form description.

    Returns:
        The referenced lemma, or None when the gloss has no usable target.
    """
    if not is_variant_gloss(gloss):
        return None

    match = re.search(
        r"(?:\bof|\bused other than(?: with)?)\s+(.+)",
        gloss,
        re.IGNORECASE,
    )

    if match is None:
        return None

    target = _TARGET_SEPARATOR.split(match.group(1), maxsplit=1)[0].strip(" '\"")

    return " ".join(target.split()) or None


def clean_gloss(
    text: str,
) -> str | None:
    """
    Clean a definition without inventing missing lexical content.

    Args:
        text: A sense definition.

    Returns:
        A definition, an empty navigation gloss, or None for damaged content.
    """
    literal = is_literal_markup(text)
    text = normalize_formatting(Attestation(text), preserve_markup=literal).text

    if not literal and (is_unrecoverable(text) or "{{" in text or "}}" in text):
        return None

    if BIBLIOGRAPHY.match(text):
        return None

    if not literal:
        text = clean_definition_references(text)

    text = normalize_formatting(Attestation(text), preserve_markup=True).text

    return normalize_definition_punctuation(text)
