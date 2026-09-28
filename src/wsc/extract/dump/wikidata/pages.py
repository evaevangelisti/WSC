"""
Read explicit Wikidata identifiers from dump definitions.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from ....constants import LANGUAGE
from ....models import POS
from ...translations import templates
from ..markup import parse_heading, section_lines

_DEFINITION = re.compile(r"^(#+)(?![#*:])\s*(.*)$")
_ETYMOLOGY = re.compile(r"Etymology (\d+)$")

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKIDATA = re.compile(r"Q[1-9]\d*")


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
