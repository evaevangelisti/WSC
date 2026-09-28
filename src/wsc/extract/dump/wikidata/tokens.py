"""
Read comparable words from dump definitions and parsed glosses.
"""

import re
from typing import cast

from ....constants import LANGUAGE
from ..markup import plain

_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TOKEN = re.compile(r"\w+", re.UNICODE)

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


def definition_tokens(
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
