"""Domain model for the wordnet the senses are aligned with."""

from dataclasses import dataclass
from enum import StrEnum

from ..pos import POS


class WordNetRelation(StrEnum):
    """
    Semantic relation directed from Wiktionary to WordNet.

    Attributes:
        EQUIVALENT: Both definitions identify the same concept.
        WIKTIONARY_NARROWER: Wiktionary identifies a strictly more specific concept.
        WIKTIONARY_BROADER: Wiktionary identifies a strictly more general concept.
    """

    EQUIVALENT = "equivalent"
    WIKTIONARY_NARROWER = "wiktionary_narrower"
    WIKTIONARY_BROADER = "wiktionary_broader"


@dataclass(frozen=True, slots=True)
class WordNetAlignment:
    """
    WordNet concept associated with a Wiktionary sense.

    Attributes:
        synset_id: Concept identifier within the extracted WordNet edition.
        relation: Relation directed from Wiktionary to WordNet.
    """

    synset_id: str
    relation: WordNetRelation


@dataclass(frozen=True, slots=True)
class Synset:
    """
    One meaning WordNet records, shared by the words that express it.

    Attributes:
        id: Identifies it within one release, such as oewn-08420278-n.
        ili: Names the same meaning across releases, such as i54321.
        pos: Its part of speech.
        definition: The gloss WordNet writes for it.
        members: Lexical forms expressing the concept.
        hypernyms: Immediate hypernym identifiers for nouns and verbs.
        examples: The sentences WordNet gives for it.
    """

    id: str
    ili: str
    pos: POS
    definition: str
    members: tuple[str, ...] = ()
    hypernyms: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
