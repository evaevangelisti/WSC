"""
Match explicit dump definitions to parsed Wiktionary senses.
"""

from difflib import SequenceMatcher

from ....identifiers import lemma_id, sense_id
from ....models import POS
from ...wiktionary.parts.glosses import clean_gloss
from ...wiktionary.schema import RawSense
from ..source_markup import MathSource
from .pages import DumpDefinition
from .tokens import definition_tokens

_MINIMUM_SIMILARITY = 0.7


def _similarity(
    written: tuple[str, ...],
    parsed: tuple[str, ...],
) -> float:
    """
    Measure how much lexical text survives Wiktextract's expansion.

    Args:
        written: Words available in the dump definition.
        parsed: Words in the corresponding parsed gloss.

    Returns:
        A match score, allowing templates to expand into longer glosses.
    """
    matcher = SequenceMatcher(None, written, parsed, autojunk=False)

    matched = sum(block.size for block in matcher.get_matching_blocks())
    ratio = matcher.ratio()

    if (
        len(written) >= 2
        and matched / len(written) >= 0.85
        and matched / len(parsed) >= 0.3
    ):
        return max(ratio, _MINIMUM_SIMILARITY + 0.2 * matched / len(parsed))

    return ratio


def _matching_sense(
    definition: DumpDefinition,
    senses: list[RawSense],
) -> RawSense | None:
    """
    Find the uniquely matching sense at the definition's own depth.

    Args:
        definition: Dump definition with an explicit Wikidata item.
        senses: Wiktextract senses under the same part of speech and etymology.

    Returns:
        The matching sense, or None when the evidence is ambiguous.
    """
    supported = [
        sense
        for sense in senses
        if all(
            identifier in sense.get("wikidata", [])
            for identifier in definition.wikidata_ids
        )
        and sense.get("glosses")
    ]

    direct = [
        sense
        for sense in supported
        if len(sense.get("glosses", [])) == definition.depth
    ]

    if len(direct) == 1:
        return direct[0]

    written = definition_tokens(definition.text)

    if not written:
        return None

    if len(supported) == 1:
        supported_glosses = supported[0].get("glosses", [])

        if (
            len(supported_glosses) > definition.depth
            and _similarity(
                written,
                definition_tokens(supported_glosses[definition.depth - 1]),
            )
            >= _MINIMUM_SIMILARITY
        ):
            return None

    candidates = direct or [
        sense for sense in senses if len(sense.get("glosses", [])) == definition.depth
    ]

    if not direct and len(supported) == 1:
        candidates = supported

    scored: list[tuple[float, RawSense]] = []

    for sense in candidates:
        glosses = sense.get("glosses", [])

        parsed = definition_tokens(glosses[-1])

        if parsed:
            scored.append((_similarity(written, parsed), sense))

    scored.sort(key=lambda item: item[0], reverse=True)

    if not scored or scored[0][0] < _MINIMUM_SIMILARITY:
        return None

    if (
        len(scored) > 1
        and scored[0][0] == scored[1][0]
        and scored[0][1].get("glosses") != scored[1][1].get("glosses")
    ):
        return None

    return scored[0][1]


def _matching_ancestor(
    definition: DumpDefinition,
    senses: list[RawSense],
) -> tuple[str, ...] | None:
    """
    Locate a definition retained only as a shared gloss ancestor.

    Args:
        definition: Dump definition with an explicit Wikidata item.
        senses: Wiktextract senses under the same part of speech and etymology.

    Returns:
        The unique ancestor gloss chain, or None if it is uncertain.
    """
    written = definition_tokens(definition.text)

    if not written:
        return None

    supported = [
        sense
        for sense in senses
        if all(
            identifier in sense.get("wikidata", [])
            for identifier in definition.wikidata_ids
        )
    ]

    candidates: list[RawSense] = supported or senses

    prefixes: set[tuple[str, ...]] = set()

    for sense in candidates:
        glosses = sense.get("glosses", [])

        if len(glosses) > definition.depth:
            prefixes.add(tuple(glosses[: definition.depth]))

    if len(prefixes) != 1:
        return None

    (prefix,) = prefixes

    if _similarity(written, definition_tokens(prefix[-1])) < _MINIMUM_SIMILARITY:
        return None

    return prefix


def _identifier_for_glosses(
    lemma: str,
    pos: POS,
    etymology: str,
    gloss_chain: tuple[str, ...],
    mathematics: tuple[MathSource, ...],
) -> str | None:
    """
    Identify a gloss chain after applying the collector's cleaning rules.

    Args:
        lemma: Headword owning the gloss chain.
        pos: Part of speech owning the gloss chain.
        etymology: Wiktextract etymology number, if present.
        gloss_chain: Raw glosses from the matched definition.
        mathematics: Original formulae from the same page.

    Returns:
        The sense identifier, or None if a gloss cannot be retained.
    """
    glosses = tuple(
        cleaned
        for gloss in gloss_chain
        if (cleaned := clean_gloss(gloss, mathematics=mathematics))
    )

    if len(glosses) != len(gloss_chain):
        return None

    return sense_id(lemma_id(lemma, pos), etymology, glosses)


def matching_identifier(
    definition: DumpDefinition,
    senses: list[RawSense],
    etymology: str,
    mathematics: tuple[MathSource, ...],
) -> str | None:
    """
    Match a definition to a direct or explicitly retained ancestor sense.

    Args:
        definition: Dump definition carrying the identifier.
        senses: Parsed senses for the entry.
        etymology: Etymology number of the parsed entry.
        mathematics: Original formulae from the same page.

    Returns:
        The matching sense identifier, or None when uncertain.
    """
    if "{{non-gloss" in definition.text.casefold():
        return None

    sense = _matching_sense(definition, senses)

    gloss_chain = (
        tuple(sense.get("glosses", []))
        if sense is not None
        else _matching_ancestor(definition, senses)
    )

    if gloss_chain is None:
        return None

    return _identifier_for_glosses(
        definition.lemma,
        definition.pos,
        etymology,
        gloss_chain,
        mathematics,
    )


def alternate_identifier(
    definition: DumpDefinition,
    senses: list[RawSense],
    etymology: str,
    mathematics: tuple[MathSource, ...],
) -> str | None:
    """
    Match an unnumbered dump sense split into a numbered parsed etymology.

    Args:
        definition: Unnumbered dump definition carrying the identifier.
        senses: Parsed senses for a numbered etymology.
        etymology: Etymology number assigned by Wiktextract.
        mathematics: Original formulae from the same page.

    Returns:
        A sense identifier only when Wiktextract retained the same item.
    """
    if "{{non-gloss" in definition.text.casefold():
        return None

    sense = _matching_sense(definition, senses)

    if sense is None or not all(
        identifier in sense.get("wikidata", [])
        for identifier in definition.wikidata_ids
    ):
        return None

    return _identifier_for_glosses(
        definition.lemma,
        definition.pos,
        etymology,
        tuple(sense.get("glosses", [])),
        mathematics,
    )
