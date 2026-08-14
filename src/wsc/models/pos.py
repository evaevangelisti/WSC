"""
The parts of speech every source is read into.
"""

from enum import StrEnum


class POS(StrEnum):
    """
    Part-of-speech tags the collector keeps.

    Values are wiktextract's own codes, so POS("adj") converts directly.
    WordNet's codes are converted onto them where they are read.
    """

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"
