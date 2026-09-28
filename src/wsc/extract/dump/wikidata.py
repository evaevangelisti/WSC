"""
Wikidata sense identifiers written in Wiktionary dump definitions.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

from ...constants import LANGUAGE
from ...files import open_compressed
from ...identifiers import lemma_id, sense_id
from ...models import POS
from ..translations import templates
from ..wiktionary.parts.glosses import clean_gloss
from ..wiktionary.schema import RawEntry, RawSense, parse_pos
from .markup import parse_heading, plain, section_lines
from .source_markup import MarkupIndex, MathSource

_DEFINITION = re.compile(r"^(#+)(?![#*:])\s*(.*)$")
_ETYMOLOGY = re.compile(r"Etymology (\d+)$")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_WIKIDATA = re.compile(r"Q[1-9]\d*")

_ANNOTATION_TEMPLATES = frozenset(
    {
        "defdate",
        "label",
        "lb",
        "q",
        "qualifier",
        "senseid",
        "senseno",
        "swp",
        "topic",
        "topics",
    }
)

_TEMPLATE_NAMES = {
    "abbr of": "abbreviation of",
    "abbreviation of": "abbreviation of",
    "alt case form of": "alternative letter case form of",
    "init of": "initialism of",
    "initialism of": "initialism of",
    "si-unit": "SI unit",
}

_MARKUP_TOKENS = frozenset({"addl", "caplc", "cont", "full", "p", "pref", "r"})

_MINIMUM_SIMILARITY = 0.7

type WikidataIds = dict[str, tuple[str, ...]]
"""
Explicit Wikidata identifiers grouped by Wiktionary sense identifier.
"""


@dataclass(frozen=True, slots=True)
class DumpDefinition:
    """
    One dump definition that explicitly names Wikidata items.

    Attributes:
        lemma: Headword owning the definition.
        pos: Part of speech of the definition.
        etymology: Numbered etymology, or an empty string.
        depth: Number of definition markers before the text.
        text: Definition text in its original markup.
        wikidata_ids: Items named on this definition alone.
    """

    lemma: str
    pos: POS
    etymology: str
    depth: int
    text: str
    wikidata_ids: tuple[str, ...]


def read_page_wikidata(
    title: str,
    markup: str,
    language_section: str,
) -> Iterator[DumpDefinition]:
    """
    Read explicit sense identifiers from one language of a dump page.

    Args:
        title: Headword named by the page.
        markup: Original page wikitext.
        language_section: Language heading to inspect.

    Yields:
        Definitions carrying a Wikidata item on their own line.
    """
    etymology = ""
    etymology_level = 0

    for pos, line in section_lines(markup, language_section):
        if pos is None:
            heading = parse_heading(line)

            if heading is None:
                continue

            level, heading_title = heading

            if level <= etymology_level:
                etymology = ""
                etymology_level = 0

            numbered_etymology = _ETYMOLOGY.fullmatch(heading_title)

            if numbered_etymology is not None:
                etymology = numbered_etymology.group(1)
                etymology_level = level

            continue

        definition = _DEFINITION.match(line)

        if definition is None:
            continue

        identifiers: list[str] = []

        for name, parameters in templates(definition.group(2)):
            value = _HTML_COMMENT.sub("", parameters.get("2", "")).strip()

            if (
                name.casefold() == "senseid"
                and parameters.get("1", "").casefold() == LANGUAGE
                and _WIKIDATA.fullmatch(value)
                and value not in identifiers
            ):
                identifiers.append(value)

        if identifiers:
            yield DumpDefinition(
                title,
                pos,
                etymology,
                len(definition.group(1)),
                definition.group(2),
                tuple(identifiers),
            )


def _tokens(
    text: str,
) -> tuple[str, ...]:
    """
    Read comparable words while dropping Wikitext annotations.

    Args:
        text: A dump definition or a parsed gloss.

    Returns:
        Case-folded words remaining after markup is removed.
    """

    def template_text(
        found: re.Match[str],
    ) -> str:
        """
        Keep lexical arguments while dropping sense annotations.

        Args:
            found: Innermost Wikitext template.

        Returns:
            Words potentially visible in the rendered gloss.
        """
        name, *arguments = found.group()[2:-2].split("|")

        name = name.casefold().strip()

        if name in _ANNOTATION_TEMPLATES:
            return " "

        if name == "place":
            return " ".join(
                argument.split("/", 1)[-1].split("=", 1)[-1]
                for argument in arguments
                if argument != LANGUAGE and not argument.startswith("tcl")
            )

        if name == "si-unit" and len(arguments) >= 4:
            return f"SI unit of {arguments[3]}"

        lexical_arguments = (
            argument for argument in arguments if argument.strip() != LANGUAGE
        )

        return " ".join((_TEMPLATE_NAMES.get(name, ""), *lexical_arguments))

    text = _HTML_COMMENT.sub(" ", text)
    previous = ""

    while previous != text:
        previous = text
        text = _TEMPLATE.sub(template_text, text)

    text = plain(_HTML.sub(" ", text))

    return tuple(
        word.casefold()
        for word in cast(list[str], _TOKEN.findall(text))
        if word.casefold() not in _MARKUP_TOKENS
    )


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

    written = _tokens(definition.text)

    if not written:
        return None

    if len(supported) == 1:
        supported_glosses = supported[0].get("glosses", [])

        if (
            len(supported_glosses) > definition.depth
            and _similarity(
                written,
                _tokens(supported_glosses[definition.depth - 1]),
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

        parsed = _tokens(glosses[-1])

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
    written = _tokens(definition.text)

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

    if _similarity(written, _tokens(prefix[-1])) < _MINIMUM_SIMILARITY:
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


def _matching_identifier(
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


def _alternate_identifier(
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
        identifier = _matching_identifier(definition, senses, etymology, mathematics)

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
        identifier = _alternate_identifier(definition, senses, etymology, mathematics)

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


def write_wikidata_ids(
    output_path: Path,
    identifiers: WikidataIds,
) -> None:
    """
    Write the explicit identifiers for later collection.

    Args:
        output_path: Cache file to create.
        identifiers: Wikidata items indexed by sense identifier.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.part")
    with open_compressed(temporary_path, "wt") as stream:
        _ = stream.write(
            json.dumps(dict(sorted(identifiers.items())), ensure_ascii=False)
        )

    _ = temporary_path.replace(output_path)


def read_wikidata_ids(
    input_path: Path,
) -> WikidataIds:
    """
    Read cached identifiers obtained from the dump.

    Args:
        input_path: Cache file created during parsing.

    Returns:
        Wikidata items indexed by sense identifier.
    """
    with open_compressed(input_path, "rt") as stream:
        contents = cast(dict[str, list[str]], json.load(stream))

    return {identifier: tuple(items) for identifier, items in contents.items()}
