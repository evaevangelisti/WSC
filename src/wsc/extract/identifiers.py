"""
How a lemma and its senses are named.
"""

from collections.abc import Iterable
from hashlib import blake2b

# Bytes of digest an identifier carries. A digest tells apart the senses of
# one headword, never the senses of the dump, so a short one is enough.
_DIGEST_SIZE = 4


def _digest(
    *written: str,
) -> str:
    """
    Name something after what it says rather than after where it was read.

    A position shifts whenever Wiktionary reorders a page or a filter changes;
    what a sense means is what survives the next dump.

    Args:
        written: The strings to name it by.

    Returns:
        The digest, in hexadecimal.
    """
    return blake2b("\n".join(written).encode(), digest_size=_DIGEST_SIZE).hexdigest()


def sense_id(
    key: str,
    glosses: Iterable[str],
) -> str:
    """
    Name one sense after the gloss chain it carries.

    Args:
        key: The headword and its part of speech, as bank.noun.
        glosses: The gloss chain, outermost first.

    Returns:
        The identifier, as bank.noun.3f9c1a2b.
    """
    return f"{key}.{_digest(*glosses)}"


def lemma_id(
    key: str,
    sense_ids: Iterable[str],
) -> str:
    """
    Name one entry after the meanings it holds.

    Args:
        key: The headword and its part of speech, as bank.noun.
        sense_ids: What each of its senses is named.

    Returns:
        The identifier, as bank.noun.7d20e4c8.
    """
    return f"{key}.{_digest(*sense_ids)}"
