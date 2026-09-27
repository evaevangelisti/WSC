"""
The little of Wikitext a translation table is written in.
"""

import re
from collections.abc import Iterator

from ...models import POS

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

_POS_BY_HEADING: dict[str, POS] = {
    "Noun": POS.NOUN,
    "Proper noun": POS.PROPN,
    "Verb": POS.VERB,
    "Adjective": POS.ADJECTIVE,
    "Adverb": POS.ADVERB,
}

_EMPHASIS = re.compile(r"'{2,5}")

_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")


def parse_heading(
    line: str,
) -> tuple[int, str] | None:
    """
    Read a section heading's depth and title.

    Args:
        line: Wikitext line to inspect.

    Returns:
        The heading depth and title, or None for other lines.
    """
    found = _HEADING.match(line)

    if found is None:
        return None

    return len(found.group(1)), found.group(2)


def section_lines(
    markup: str,
    language_section: str,
) -> Iterator[tuple[POS | None, str]]:
    """
    Walk page lines with the part of speech active in one language.

    Args:
        markup: Original page wikitext.
        language_section: Language heading to inspect.

    Yields:
        The active part of speech and line, or None at a heading.
    """
    reading_language = False

    pos: POS | None = None
    pos_level = 0

    for line in markup.splitlines():
        heading = parse_heading(line)

        if heading is None:
            if reading_language and pos is not None:
                yield pos, line

            continue

        level, title = heading

        if level == 2:
            reading_language = title == language_section

        if level <= pos_level:
            pos = None

        found_pos = _POS_BY_HEADING.get(title)

        if reading_language and found_pos is not None:
            pos = found_pos
            pos_level = level

        yield None, line


def plain(
    text: str,
) -> str:
    """
    Read the text a piece of markup shows, so two sources spell a gloss alike.

    Args:
        text: The markup.

    Returns:
        The same text, its emphasis dropped and its links reduced to labels.
    """
    return _EMPHASIS.sub("", _LINK.sub(r"\1", text)).strip()
