"""The meanings an entry holds."""

from ....identifiers import sense_id
from ....models import Sense
from ..schema import RawSense
from .glosses import clean_gloss, is_form_gloss, is_synonym_gloss, is_variant_gloss
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

        original_glosses = tuple(
            gloss.strip() for gloss in raw_sense.get("glosses", []) if gloss.strip()
        )

        if not original_glosses or any(
            is_form_gloss(gloss) or is_synonym_gloss(gloss) or is_variant_gloss(gloss)
            for gloss in original_glosses
        ):
            continue

        cleaned_glosses = [clean_gloss(gloss) for gloss in original_glosses]

        if any(gloss is None for gloss in cleaned_glosses):
            continue

        glosses = tuple(gloss for gloss in cleaned_glosses if gloss)

        if not glosses:
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
