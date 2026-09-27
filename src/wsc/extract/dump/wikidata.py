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

_DEFINITION = re.compile(r"^(#+)(?![#*:])\s*(.*)$")
_ETYMOLOGY = re.compile(r"Etymology (\d+)$")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
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
            value = parameters.get("2", "").strip()

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

    def template_text(found: re.Match[str]) -> str:
        """
        Keep lexical arguments while dropping sense annotations.

        Args:
            found: Innermost Wikitext template.

        Returns:
            Words potentially visible in the rendered gloss.
        """
        name, *arguments = found.group()[2:-2].split("|")

        if name.casefold().strip() in _ANNOTATION_TEMPLATES:
            return " "

        lexical_arguments = (
            argument for argument in arguments if argument.strip() != LANGUAGE
        )

        return " ".join(
            (_TEMPLATE_NAMES.get(name.casefold().strip(), ""), *lexical_arguments)
        )

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
    written = _tokens(definition.text)

    if not written:
        return None

    scored: list[tuple[float, RawSense]] = []

    for sense in senses:
        glosses = sense.get("glosses", [])

        if len(glosses) != definition.depth:
            continue

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


def index_wikidata_ids(
    definitions: Iterable[DumpDefinition],
    entries: Iterable[RawEntry],
) -> tuple[WikidataIds, int]:
    """
    Associate explicit dump identifiers with parsed sense identities.

    Args:
        definitions: Definitions read directly from dump markup.
        entries: Parsed entries used to locate the corresponding glosses.

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

    for entry in entries:
        if entry.get("lang_code") != LANGUAGE:
            continue

        lemma = entry.get("word", "").strip()

        try:
            pos = parse_pos(entry.get("pos", ""))
        except ValueError:
            continue

        etymology = str(entry.get("etymology_number", ""))
        definitions_for_entry = by_entry.get((lemma, pos, etymology), ())

        for definition in definitions_for_entry:
            sense = _matching_sense(definition, entry.get("senses", []))

            if sense is None:
                continue

            glosses = tuple(
                cleaned
                for gloss in sense.get("glosses", [])
                if (cleaned := clean_gloss(gloss))
            )

            if len(glosses) != len(sense.get("glosses", [])):
                continue

            identifier = sense_id(lemma_id(lemma, pos), etymology, glosses)
            identifiers = indexed.setdefault(identifier, [])

            for wikidata_id in definition.wikidata_ids:
                if wikidata_id not in identifiers:
                    identifiers.append(wikidata_id)

            matched_definitions.add(id(definition))

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
