"""Locating a lemma inside the sentences that attest it."""

import re
from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import tee

from kwic import POS as UNIVERSAL_POS
from kwic import Locator, Query

from ..constants import BATCH_SIZE, PROCESSES, SPACY_PIPELINE
from ..models import POS, Engine, Offset

# Engine outputs use Universal Dependencies part-of-speech tags.
_UNIVERSAL_TAGS = {
    POS.NOUN: UNIVERSAL_POS.NOUN,
    POS.PROPN: UNIVERSAL_POS.PROPN,
    POS.VERB: UNIVERSAL_POS.VERB,
    POS.ADJECTIVE: UNIVERSAL_POS.ADJ,
    POS.ADVERB: UNIVERSAL_POS.ADV,
}

# Lookarounds preserve boundaries for forms containing apostrophes and hyphens.
_BOUNDED = r"(?<!\w)(?:{alternation})(?!\w)"


def open_locator(
    engine: Engine,
    processes: int = PROCESSES,
    batch_size: int = BATCH_SIZE,
    *,
    gpu: bool = False,
) -> Locator:
    """
    Load the search the sentences are read by.

    Stanza runs one process whatever it is asked for.

    Args:
        engine: Which analyser to read with.
        processes: How many processes it may run.
        batch_size: How many sentences it takes at a time.
        gpu: Whether to require GPU inference.

    Returns:
        The search, its model loaded on the first sentence.
    """
    match engine:
        case Engine.SPACY:
            from kwic import SpacyEngine

            return Locator(SpacyEngine(SPACY_PIPELINE, batch_size, processes, gpu=gpu))

        case Engine.STANZA:
            from kwic.engines.stanza import StanzaEngine

            return Locator(StanzaEngine(batch_size=batch_size, gpu=gpu))

        case Engine.LEMMINFLECT:
            from kwic.engines.lemminflect import LemmInflectEngine

            return Locator(
                LemmInflectEngine(
                    batch_size=batch_size,
                    processes=processes,
                    gpu=gpu,
                )
            )


def build_query(
    lemma: str,
    pos: POS,
    forms: frozenset[str],
) -> Query:
    """
    Say what is to be looked for in the sentences of one entry.

    The part of speech narrows a one-word lemma, and the listed forms take an occurrence
    the engine lemmatised as something else.

    Args:
        lemma: The headword.
        pos: Its part of speech.
        forms: The shapes it takes, the headword among them.

    Returns:
        The query every sentence of the entry is read for.
    """
    return Query(lemma, _UNIVERSAL_TAGS[pos], forms)


@lru_cache(maxsize=1)
def _compile_forms(
    forms: frozenset[str],
) -> re.Pattern[str]:
    """
    Compile the forms of one lemma into the pattern they are matched by.

    Sentences are searched an entry at a time, so holding the last pattern alone
    compiles once per entry.

    Args:
        forms: The headword and its inflections.

    Returns:
        The pattern matching any of them, whatever the case.
    """
    # Longest-first matching preserves multiword forms; lexical sorting stabilizes ties.
    alternation = "|".join(
        re.escape(form) for form in sorted(forms, key=lambda form: (-len(form), form))
    )

    return re.compile(_BOUNDED.format(alternation=alternation), re.IGNORECASE)


def match_forms(
    text: str,
    forms: frozenset[str],
) -> tuple[Offset, ...]:
    """
    Match the listed forms in one sentence, letter for letter and case aside.

    Args:
        text: The sentence to search.
        forms: The headword and its inflections.

    Returns:
        The ranges they occupy, leftmost first and never overlapping.
    """
    return tuple(match.span() for match in _compile_forms(forms).finditer(text))


def find_word_offsets(
    locator: Locator,
    searches: Iterable[tuple[str, Query]],
) -> Iterator[tuple[Offset, ...]]:
    """
    Locate the lemma of every sentence handed over.

    A reading tells a word from its homograph and takes an inflection nobody listed; the
    listed forms are matched only where it found nothing at all.

    Args:
        locator: The search to read with.
        searches: A sentence and the lemma to look for in it, in pairs.

    Yields:
        Lemma ranges in sentence order, sorted from left to right.
    """
    reading_searches, matching_searches = tee(searches)

    queried = ((text, (query,)) for text, query in reading_searches)

    for matches, (text, query) in zip(
        locator.find_all(queried),
        matching_searches,
        strict=True,
    ):
        # String-based sentence queries always provide character offsets.
        word_offsets = tuple(
            match.offsets for match in matches if match.offsets is not None
        )

        yield word_offsets or match_forms(text, query.forms)
