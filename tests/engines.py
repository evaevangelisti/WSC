"""
The engine the suite reads with, so that no property waits on a model.

How a real pipeline tags and lemmatises is kwic's to test; what the suite
needs is a reading that never varies.
"""

import re
from collections.abc import Iterable, Iterator
from typing import override

from kwic import POS, Context, Engine, Token

_WORD = re.compile(r"\S+")


class WhitespaceEngine(Engine):
    """
    Cuts a context on whitespace and reads every word as the noun it spells.
    """

    @override
    def analyse_all(
        self,
        contexts: Iterable[Context],
    ) -> Iterator[tuple[Token, ...]]:
        """
        Read every context into the words whitespace parts it into.

        Args:
            contexts: The texts to read, or the words they were split into.

        Yields:
            The words of one context, in the order the contexts came in.
        """
        for context in contexts:
            yield tuple(self._read(context))

    @staticmethod
    def _read(
        context: Context,
    ) -> Iterator[Token]:
        """
        Read one context, giving a range only where there is a text to index.

        Args:
            context: The text to read, or the words it was split into.

        Yields:
            The words it is made of, in the order they are written.
        """
        if isinstance(context, str):
            for match in _WORD.finditer(context):
                form = match.group()

                yield Token(form, POS.NOUN, form, match.span())

            return

        for form in context:
            yield Token(form, POS.NOUN, form, None)
