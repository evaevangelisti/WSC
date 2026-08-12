"""
Locating a lemma inside the sentences that attest it.
"""

import re
from functools import lru_cache

from ..models import WordOffset

# A form stands on its own, never inside a longer word. \b would fall on the
# wrong side of one opening or closing with an apostrophe or a hyphen.
_BOUNDED = r"(?<!\w)(?:{alternation})(?!\w)"


@lru_cache(maxsize=1)
def _compile_forms(
    forms: frozenset[str],
) -> re.Pattern[str]:
    """
    Compile the forms of one lemma into the pattern they are read by.

    Entries are read one at a time, and every sentence of one against the same
    forms, so holding the last pattern alone compiles once per entry.

    Args:
        forms: The headword and its inflections.

    Returns:
        The pattern matching any of them, whatever the case.
    """
    # The longest form first, so that give up wins over give. Ties are broken
    # alphabetically, a frozenset having no order of its own.
    alternation = "|".join(
        re.escape(form) for form in sorted(forms, key=lambda form: (-len(form), form))
    )

    return re.compile(_BOUNDED.format(alternation=alternation), re.IGNORECASE)


def find_word_offsets(
    text: str,
    forms: frozenset[str],
) -> tuple[WordOffset, ...]:
    """
    Locate every occurrence of a lemma in one sentence.

    Args:
        text: The sentence to read.
        forms: The headword and its inflections.

    Returns:
        The ranges it occupies, leftmost first and never overlapping.
    """
    return tuple(
        (match.start(), match.end()) for match in _compile_forms(forms).finditer(text)
    )
