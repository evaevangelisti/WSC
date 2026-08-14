"""
Domain model for the wordnet the senses are aligned with.
"""

from dataclasses import dataclass

from ..pos import POS


@dataclass(frozen=True, slots=True)
class Synset:
    """
    One meaning WordNet records, shared by the words that express it.

    Attributes:
        id: Identifies it within one release, such as oewn-08420278-n.
        ili: Names the same meaning across releases, such as i54321.
        pos: Its part of speech.
        definition: The gloss WordNet writes for it.
        members: The words expressing it, which is what makes it a candidate
        for a lemma.
        hypernyms: The synsets it is a kind of, by identifier. Nouns and verbs
        alone are arranged that way.
        examples: The sentences WordNet gives for it.
    """

    id: str
    ili: str
    pos: POS
    definition: str
    members: tuple[str, ...] = ()
    hypernyms: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
