"""Classify glosses that describe forms, variants, and synonyms."""

import re

_FORM_PREFIXES = frozenset(
    {
        "accusative of",
        "accusative singular of",
        "alternative plural of",
        "animate accusative plural",
        "animate accusative singular",
        "a plural of",
        "attributive form",
        "comparative degree of",
        "comparative form of",
        "dative of",
        "dative plural",
        "dative/prepositional singular of",
        "dative singular of",
        "definite nominative/accusative singular",
        "feminine plural",
        "feminine singular",
        "first-person plural",
        "first-person singular future",
        "first-person singular non-past",
        "first-person singular past",
        "first-person singular present",
        "genitive/accusative plural of",
        "genitive/accusative/prepositional plural of",
        "genitive/accusative singular of",
        "genitive/dative/prepositional singular",
        "genitive/dative/prepositional singular of",
        "genitive of",
        "genitive plural",
        "genitive/prepositional plural of",
        "genitive singular of",
        "imperfective and stem (or continuative) forms of",
        "inanimate accusative plural",
        "inflection of",
        "instrumental plural of",
        "instrumental singular",
        "locative singular of",
        "masculine plural",
        "masculine singular past",
        "neuter singular past",
        "nominative/accusative plural",
        "nominative plural",
        "past of",
        "past participle",
        "past tense of",
        "plural of",
        "plural past",
        "prepositional plural of",
        "prepositional singular of",
        "present participle of",
        "present participle and",
        "second-person",
        "second-person feminine",
        "second-person masculine",
        "second-person plural",
        "second-person singular simple",
        "second-person singular simple past",
        "second-person singular simple present",
        "short feminine singular",
        "short masculine singular",
        "short neuter singular",
        "short plural of",
        "short plural past",
        "simple past of",
        "simple past and",
        "simple past tense",
        "simple past tense and",
        "simple past tense of",
        "stem or continuative form of",
        "superlative form",
        "third-person feminine",
        "third-person masculine",
        "third-person plural",
        "third-person singular",
        "third-person singular simple",
        "third-person singular simple present",
    }
)

_SYNONYM_PREFIXES = frozenset({"synonym", "synonym of"})

_VARIANT_PREFIXES = frozenset(
    {
        "abbreviation",
        "abbreviation of",
        "active participle of",
        "a diminutive of",
        "alternative form",
        "alternative form of",
        "alternative letter-case",
        "alternative letter-case form",
        "alternative letter-case form of",
        "alternative spelling of",
        "an aphetic form of",
        "anglicized form of",
        "aphetic form of",
        "archaic",
        "archaic form",
        "archaic form of",
        "archaic spelling",
        "archaic spelling of",
        "a transliteration",
        "dated form",
        "dated form of",
        "elative degree of",
        "elongated form of",
        "eye dialect",
        "eye dialect spelling",
        "eye dialect spelling of",
        "informal form of",
        "initialism of",
        "misspelling",
        "misspelling of",
        "non-oxford",
        "non-oxford british",
        "non-oxford british english",
        "non-oxford british english standard",
        "nonstandard form of",
        "nonstandard spelling of",
        "obsolete",
        "obsolete spelling",
        "obsolete spelling of",
        "passive participle of",
        "pronunciation spelling of",
        "rare form of",
        "rare spelling",
        "rare spelling of",
        "reduced form of",
        "representing a pronunciation of",
        "shortened form of",
        "short form of",
        "singulative of",
        "the condition",
        "this term needs a definition. please help out and",
        "this term needs a translation to english",
        "uncommon form of",
        "unstressed form of",
        "used other",
        "used other than",
        "used other than with",
        "verbal noun of",
    }
)

_TARGET_SEPARATOR = re.compile(r"\s*(?:[;,]|\(|$)")


def _normalize(
    gloss: str,
) -> str:
    return " ".join(gloss.casefold().strip().split())


def _matches_category(
    gloss: str,
    categories: frozenset[str],
) -> bool:
    normalized = _normalize(gloss)
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in categories
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
    return _matches_category(gloss, _FORM_PREFIXES)


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
    return _matches_category(gloss, _SYNONYM_PREFIXES)


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
    return _matches_category(gloss, _VARIANT_PREFIXES)


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
