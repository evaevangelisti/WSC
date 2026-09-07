"""The analysers a sentence may be read with."""

from enum import StrEnum


class Engine(StrEnum):
    """
    What a sentence is read with when the lemma is looked for in it.

    The selected engine determines sentence analysis accuracy and runtime.
    """

    SPACY = "spacy"
    STANZA = "stanza"
    LEMMINFLECT = "lemminflect"
