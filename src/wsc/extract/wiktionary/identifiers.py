"""How a lemma and its senses are named."""

from collections.abc import Iterable
from hashlib import blake2b

from ...models import POS

_DIGEST_SIZE = 4


def _digest(
    *written: str,
) -> str:
    """
    Generate a stable identifier from source content.

    Args:
        written: The strings to name it by.

    Returns:
        The digest, in hexadecimal.
    """
    return blake2b("\n".join(written).encode(), digest_size=_DIGEST_SIZE).hexdigest()


def lemma_id(
    lemma: str,
    pos: POS,
) -> str:
    """
    Name one entry after the headword and part of speech gathering it.

    Args:
        lemma: The headword.
        pos: Its part of speech.

    Returns:
        The identifier, as bank.noun.
    """
    return f"{lemma}.{pos}"


def sense_id(
    lemma_id: str,
    etymology: str,
    glosses: Iterable[str],
) -> str:
    """
    Name one sense after its gloss chain and the etymology holding it.

    Etymology distinguishes senses sharing a gloss chain.

    Args:
        lemma_id: What the entry is named.
        etymology: Etymology identifier, empty for entries with one etymology.
        glosses: The gloss chain, outermost first.

    Returns:
        The identifier, as bank.noun.3f9c1a2b.
    """
    return f"{lemma_id}.{_digest(etymology, *glosses)}"
