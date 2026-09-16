"""Candidate construction independent of model predictions."""

from collections import defaultdict
from collections.abc import Iterable

from ..models import POS, Lemma, Synset


class WordNetCandidates:
    """WordNet candidates retrieved by lexical form and part of speech."""

    def __init__(
        self,
        synsets: Iterable[Synset],
    ) -> None:
        """
        Index synsets under all their lexical members.

        Args:
            synsets: Cached WordNet concepts.
        """
        self._members: defaultdict[tuple[str, POS], dict[str, Synset]] = defaultdict(
            dict
        )

        for synset in synsets:
            for member in synset.members:
                self._members[self._normalize(member), synset.pos][synset.id] = synset

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize lexical lookup without changing candidate definitions.

        Args:
            text: Headword or WordNet member.

        Returns:
            Case-folded words separated by spaces.
        """
        return " ".join(text.replace("_", " ").casefold().split())

    def candidates(
        self,
        lemma: Lemma,
    ) -> tuple[Synset, ...]:
        """
        Retrieve candidates including spelling variants and nominal proper names.

        Args:
            lemma: Entry whose WordNet senses are requested.

        Returns:
            Unique candidates sorted by synset identifier.
        """
        pos = POS.NOUN if lemma.pos == POS.PROPN else lemma.pos

        found_synsets: dict[str, Synset] = {}

        for form in (lemma.lemma, *sorted(lemma.variants)):
            found_synsets.update(self._members.get((self._normalize(form), pos), {}))

        return tuple(found_synsets[identifier] for identifier in sorted(found_synsets))

    def synonyms(
        self,
        lemma: Lemma,
        synset: Synset,
    ) -> tuple[str, ...]:
        """
        Return synset members other than the queried lemma.

        Args:
            lemma: Entry whose candidates are being described.
            synset: Candidate WordNet concept.

        Returns:
            Other lexical members, retaining WordNet order.
        """
        return tuple(
            member
            for member in synset.members
            if self._normalize(member) != self._normalize(lemma.lemma)
        )
