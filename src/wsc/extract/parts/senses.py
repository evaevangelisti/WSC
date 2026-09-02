"""
The meanings an entry holds.
"""

from ...models import Sense
from ..identifiers import sense_id
from ..schema import RawSense
from .sentences import parse_sentences

# Tags marking a sense that states a form rather than a meaning, which is
# a third of what Wiktionary writes.
_FORM_TAGS = frozenset({"form-of", "alt-of"})


def parse_senses(
    raw_senses: list[RawSense],
    key: str,
    minimum_year: int | None,
    maximum_year: int | None,
) -> list[Sense]:
    """
    Collect the senses of one entry.

    A nested sense is kept alongside its parent, which often carries
    examples of its own. A sense stating a form names no meaning.

    Args:
        raw_senses: What wiktextract listed under the entry.
        key: The headword and its part of speech, opening each identifier.
        minimum_year: Oldest quotation to keep, or None for no bound.
        maximum_year: Newest quotation to keep, or None for no bound.

    Returns:
        The senses that carry at least one gloss and define something.
    """
    senses: list[Sense] = []

    for raw_sense in raw_senses:
        tags = tuple(raw_sense.get("tags", []))
        if not _FORM_TAGS.isdisjoint(tags):
            continue

        glosses = tuple(
            gloss.strip() for gloss in raw_sense.get("glosses", []) if gloss.strip()
        )
        if not glosses:
            continue

        senses.append(
            Sense(
                sense_id(key, glosses),
                glosses,
                tuple(raw_sense.get("topics", [])),
                tags,
                parse_sentences(
                    raw_sense.get("examples", []),
                    minimum_year,
                    maximum_year,
                ),
                tuple(raw_sense.get("senseid", [])),
                tuple(raw_sense.get("wikidata", [])),
            )
        )

    return senses
