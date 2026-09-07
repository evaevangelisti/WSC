"""The little of Wikitext a translation table is written in."""

import re

_EMPHASIS = re.compile(r"'{2,5}")

_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")

# Gloss values can contain equals signs.
_NAMED = re.compile(r"^([A-Za-z0-9_-]+)=(.*)$", re.DOTALL)


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


def arguments(
    body: str,
) -> tuple[list[str], dict[str, str]]:
    """
    Split what stands between a template's name and its closing braces.

    Args:
        body: The arguments, still parted by pipes.

    Returns:
        The positional arguments in order, and the named ones by name.
    """
    positional_arguments: list[str] = []
    named_arguments: dict[str, str] = {}

    for argument in body.split("|"):
        found_name = _NAMED.match(argument.strip())

        if found_name:
            named_arguments[found_name.group(1)] = found_name.group(2).strip()
        else:
            positional_arguments.append(argument.strip())

    return positional_arguments, named_arguments
