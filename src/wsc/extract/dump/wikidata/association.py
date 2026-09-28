"""
Associate dump Wikidata identifiers with parsed sense identities.
"""

from collections import defaultdict
from collections.abc import Iterable

from ....constants import LANGUAGE
from ....models import POS
from ...wiktionary.schema import RawEntry, RawSense, parse_pos
from ..source_markup import MarkupIndex, MathSource
from .cache import WikidataIds, read_wikidata_ids, write_wikidata_ids
from .matching import alternate_identifier, matching_identifier
from .pages import DumpDefinition, read_page_wikidata

__all__ = [
    "DumpDefinition",
    "WikidataIds",
    "index_wikidata_ids",
    "read_page_wikidata",
    "read_wikidata_ids",
    "write_wikidata_ids",
]


def _append_identifiers(
    indexed: dict[str, list[str]],
    identifier: str,
    wikidata_ids: tuple[str, ...],
) -> None:
    """
    Record each explicit item once for a matched sense.

    Args:
        indexed: Mutable map of sense identifiers to Wikidata items.
        identifier: Sense to which the dump definition belongs.
        wikidata_ids: Items named directly on the dump definition.
    """
    identifiers = indexed.setdefault(identifier, [])

    for wikidata_id in wikidata_ids:
        if wikidata_id not in identifiers:
            identifiers.append(wikidata_id)


def _record_direct_definitions(
    definitions: Iterable[DumpDefinition],
    senses: list[RawSense],
    etymology: str,
    indexed: dict[str, list[str]],
    matched: set[int],
    mathematics: tuple[MathSource, ...],
) -> None:
    """
    Index dump definitions under their matching Wiktextract etymology.

    Args:
        definitions: Definitions from the same entry and etymology.
        senses: Parsed senses of that entry.
        etymology: Number of the parsed etymology, if any.
        indexed: Mutable sense-to-item map.
        matched: Identities of definitions already associated.
        mathematics: Original formulae from the same page.
    """
    for definition in definitions:
        identifier = matching_identifier(definition, senses, etymology, mathematics)

        if identifier is not None:
            _append_identifiers(indexed, identifier, definition.wikidata_ids)
            matched.add(id(definition))


def _record_alternate_definitions(
    definitions: Iterable[DumpDefinition],
    senses: list[RawSense],
    etymology: str,
    matches: defaultdict[int, set[str]],
    originals: dict[int, DumpDefinition],
    mathematics: tuple[MathSource, ...],
) -> None:
    """
    Collect uniquely checkable matches across changed etymology numbering.

    Args:
        definitions: Unnumbered dump definitions for this headword and POS.
        senses: Parsed senses from a numbered etymology.
        etymology: Number assigned by Wiktextract.
        matches: Candidate parsed sense identifiers by dump definition.
        originals: Dump definitions indexed by object identity.
        mathematics: Original formulae from the same page.
    """
    for definition in definitions:
        identifier = alternate_identifier(definition, senses, etymology, mathematics)

        if identifier is not None:
            matches[id(definition)].add(identifier)
            originals[id(definition)] = definition


def index_wikidata_ids(
    definitions: Iterable[DumpDefinition],
    entries: Iterable[RawEntry],
    markup_index: MarkupIndex | None = None,
) -> tuple[WikidataIds, int]:
    """
    Associate explicit dump identifiers with parsed sense identities.

    Args:
        definitions: Definitions read directly from dump markup.
        entries: Parsed entries used to locate the corresponding glosses.
        markup_index: Original formulae indexed by source page.

    Returns:
        Sense identifiers and the number of definitions left unmatched.
    """
    by_entry: defaultdict[tuple[str, POS, str], list[DumpDefinition]] = defaultdict(
        list,
    )

    for definition in definitions:
        by_entry[(definition.lemma, definition.pos, definition.etymology)].append(
            definition
        )

    indexed: dict[str, list[str]] = {}
    matched_definitions: set[int] = set()

    alternate_matches: defaultdict[int, set[str]] = defaultdict(set)
    alternate_definitions: dict[int, DumpDefinition] = {}

    for entry in entries:
        if entry.get("lang_code") != LANGUAGE:
            continue

        lemma = entry.get("word", "").strip()

        try:
            pos = parse_pos(entry.get("pos", ""))
        except ValueError:
            continue

        etymology = str(entry.get("etymology_number", ""))
        page_markup = markup_index.get(lemma) if markup_index is not None else None
        mathematics = page_markup.mathematics if page_markup else ()
        definitions_for_entry = by_entry.get((lemma, pos, etymology), ())

        _record_direct_definitions(
            definitions_for_entry,
            entry.get("senses", []),
            etymology,
            indexed,
            matched_definitions,
            mathematics,
        )

        if etymology:
            _record_alternate_definitions(
                by_entry.get((lemma, pos, ""), ()),
                entry.get("senses", []),
                etymology,
                alternate_matches,
                alternate_definitions,
                mathematics,
            )

    for definition_id, candidates in alternate_matches.items():
        if definition_id in matched_definitions or len(candidates) != 1:
            continue

        (identifier,) = candidates
        _append_identifiers(
            indexed,
            identifier,
            alternate_definitions[definition_id].wikidata_ids,
        )
        matched_definitions.add(definition_id)

    return (
        {identifier: tuple(items) for identifier, items in indexed.items()},
        sum(len(group) for group in by_entry.values()) - len(matched_definitions),
    )
