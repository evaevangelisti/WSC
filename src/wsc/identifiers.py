"""Name lexical entries, senses, translation tables, and alignment queries."""

from collections.abc import Iterable
from hashlib import blake2b

from .models import POS
from .models.alignment import AlignmentTask

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


def translation_table_id(
    lemma_id: str,
    gloss: str,
) -> str:
    """
    Name a translation table after its owning lemma and gloss.

    Args:
        lemma_id: What the entry is named.
        gloss: The table's meaning heading.

    Returns:
        The identifier, as bank.noun.tr.3f9c1a2b.
    """
    return f"{lemma_id}.tr.{_digest(gloss)}"


def query_id(
    task: AlignmentTask,
    identifier: str,
) -> str:
    """Build a task-scoped identifier for an alignment query."""
    return f"{task.value}:{identifier}"
