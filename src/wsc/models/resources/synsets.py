"""Domain models for the generic synsets used in alignment."""

from dataclasses import dataclass
from enum import StrEnum

from ..pos import POS


class SynsetRelation(StrEnum):
    """Semantic relation directed from Wiktionary to a synset."""

    EQUIVALENT = "equivalent"
    WIKTIONARY_NARROWER = "wiktionary_narrower"
    WIKTIONARY_BROADER = "wiktionary_broader"


@dataclass(frozen=True, slots=True)
class SynsetAlignment:
    """Synset associated with a Wiktionary sense."""

    synset_id: str
    relation: SynsetRelation
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SynsetMember:
    """One lexical member of a synset."""

    lemma: str
    source: str = ""


@dataclass(frozen=True, slots=True)
class Synset:
    """One general lexical concept shared by one or more members."""

    id: str
    pos: POS
    members: tuple[SynsetMember, ...]
    glosses: tuple[str, ...]
    examples: tuple[str, ...] = ()
