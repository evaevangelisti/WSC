"""
Domain models for the generic synsets used in alignment.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..pos import POS


class SynsetRelation(StrEnum):
    """
    Semantic relation directed from Wiktionary to a synset.
    """

    EQUIVALENT = "equivalent"
    WIKTIONARY_NARROWER = "wiktionary_narrower"
    WIKTIONARY_BROADER = "wiktionary_broader"


@dataclass(frozen=True, slots=True)
class SynsetAlignment:
    """
    Synset associated with a Wiktionary sense.

    Attributes:
        synset_id: Identifier of the associated synset.
        relation: Semantic relation directed from Wiktionary.
        sources: Resources supporting the queried member.
    """

    synset_id: str
    relation: SynsetRelation
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SynsetResource:
    """
    Describe one source's evidence for a synset.

    Attributes:
        members: Lemmas belonging to the synset in this resource.
        glosses: Descriptions supplied by this resource.
        examples: Usage examples supplied by this resource.
    """

    members: tuple[str, ...]
    glosses: tuple[str, ...]
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Synset:
    """
    One lexical concept supported by named resources.

    Attributes:
        id: Stable identifier shared across resources.
        pos: Part of speech shared by the source records.
        resources: Lexical evidence indexed by resource name.
    """

    id: str
    pos: POS
    resources: Mapping[str, SynsetResource]
