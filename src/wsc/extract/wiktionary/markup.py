"""
The wiki markup wiktextract now and then leaves unexpanded.
"""

import re

# A template or a link, both known by the pipe that parts their arguments.
# The pipe is what tells them from what a text writes for itself: an editorial
# ellipsis reads [[…]], a nested aside [P[eter] S[imon]], and a page quoting
# a template language reads {{ post.title|title }}, none of them markup.
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
