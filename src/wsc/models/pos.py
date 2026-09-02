"""
The parts of speech every source is read into.
"""

from enum import StrEnum


class POS(StrEnum):
    """
    Part-of-speech tags the collector keeps.

    Values are wiktextract's own codes, so POS("adj") converts directly. A
    proper noun stands apart from a common one, as Wiktionary writes it.
    """

    NOUN = "noun"
    NAME = "name"
    VERB = "verb"
    ADJECTIVE = "adj"
    ADVERB = "adv"
