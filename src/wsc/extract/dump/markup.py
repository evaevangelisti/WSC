"""The little of Wikitext a translation table is written in."""

import re

_EMPHASIS = re.compile(r"'{2,5}")

_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")


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
