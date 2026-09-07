"""The wiki markup wiktextract now and then leaves unexpanded."""

import re

# Markup matching preserves literal brackets and template-language examples.
_MARKUP = re.compile(r"\{\{[a-z][a-z0-9-]*\||\{\{\||\[\[[^\]|]*\|")


def carries_markup(
    text: str,
) -> bool:
    """
    Say whether a text carries markup that was meant to be expanded away.

    Args:
        text: The gloss or sentence to read.

    Returns:
        Whether a template or a link is left in it.
    """
    return _MARKUP.search(text) is not None
