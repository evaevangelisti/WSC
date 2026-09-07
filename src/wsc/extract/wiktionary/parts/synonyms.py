"""Other words standing for what a sense means."""

from ..schema import RawSynonym


def parse_synonyms(
    raw_synonyms: list[RawSynonym],
    lemma: str,
) -> tuple[str, ...]:
    """
    Read the synonyms wiktextract tied to one sense.

    Args:
        raw_synonyms: What wiktextract listed under the sense.
        lemma: The headword, which is no synonym of itself.

    Returns:
        The words offered for that meaning alone, a repeat kept once.
    """
    seen: dict[str, None] = {}

    for raw_synonym in raw_synonyms:
        word = raw_synonym.get("word", "").strip()

        if word and word != lemma:
            seen[word] = None

    return tuple(seen)
