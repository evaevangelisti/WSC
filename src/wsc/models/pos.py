"""The parts of speech every source is read into."""

from enum import StrEnum


class POS(StrEnum):
    """
    Part-of-speech tags.
    """

    NOUN = "noun"
    PROPN = "propn"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"
