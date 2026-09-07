"""The meanings an entry holds."""

from ....models import Sense
from ..identifiers import sense_id
from ..markup import carries_markup
from ..schema import RawSense
from .sentences import parse_sentences
from .synonyms import parse_synonyms

_FORM_TAGS = frozenset({"form-of", "alt-of"})


def parse_senses(
    raw_senses: list[RawSense],
    lemma_id: str,
    lemma: str,
    etymology: str,
    minimum_year: int | None,
    maximum_year: int | None,
) -> list[Sense]:
    """
    Collect the senses of one entry.

    Args:
        raw_senses: What wiktextract listed under the entry.
        lemma_id: What the entry is named, opening each identifier.
        lemma: The headword, which is no synonym of itself.
        etymology: Which etymology of the entry the senses sit under.
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

        if not glosses or any(carries_markup(gloss) for gloss in glosses):
            continue

        senses.append(
            Sense(
                sense_id(lemma_id, etymology, glosses),
                glosses,
                etymology,
                parse_synonyms(raw_sense.get("synonyms", []), lemma),
                tuple(raw_sense.get("topics", [])),
                tags,
                parse_sentences(
                    raw_sense.get("examples", []),
                    minimum_year,
                    maximum_year,
                ),
                tuple(raw_sense.get("wikidata", [])),
            )
        )

    return senses
