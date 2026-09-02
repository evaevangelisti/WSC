"""
The analysers a sentence may be read with.
"""

from enum import StrEnum


class Engine(StrEnum):
    """
    What a sentence is read with when the lemma is looked for in it.

    A sentence is analysed rather than searched, so a run is as right as the
    reading behind it, and as slow.
    """

    SPACY = "spacy"
    STANZA = "stanza"
    LEMMINFLECT = "lemminflect"
