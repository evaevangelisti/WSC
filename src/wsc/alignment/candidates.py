"""
Index synset candidates and select source evidence for each pass.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from copy import copy
from urllib.parse import quote

from ..models import POS, Lemma, Synset, SynsetRelation, SynsetResource


class SynsetCandidates:
    """
    Retrieve synsets by lexical form and select resource evidence.
    """

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize lexical lookup without changing candidate definitions.

        Args:
            text: Headword or synset member.

        Returns:
            Case-folded words separated by spaces.
        """
        return " ".join(text.replace("_", " ").casefold().split())

    def __init__(
        self,
        synsets: Iterable[Synset],
    ) -> None:
        """
        Index synsets under members from every resource.

        Args:
            synsets: Concepts supplied for alignment.
        """
        self._members: defaultdict[tuple[str, POS], dict[str, Synset]] = defaultdict(
            dict,
        )

        self._available_sources: set[str] = set()
        self._previous_sources: tuple[str, ...] = ()

        self._stage: str = ""
        self._cache_stage: str = ""

        self._skip_equivalent: bool = False

        for synset in synsets:
            self._available_sources.update(synset.resources)

            for resource in synset.resources.values():
                for member in resource.members:
                    self._members[self._normalize(member), synset.pos][synset.id] = (
                        synset
                    )

    @property
    def available_sources(
        self,
    ) -> frozenset[str]:
        """
        Return resource names declared by the indexed synsets.
        """
        return frozenset(self._available_sources)

    @property
    def stage(
        self,
    ) -> str:
        """
        Return the cache and prompt label for this pass.
        """
        return self._stage

    @property
    def cache_stage(
        self,
    ) -> str:
        """
        Return a query label distinguishing ordered resource selections.
        """
        return self._cache_stage

    @property
    def skip_equivalent(
        self,
    ) -> bool:
        """
        Return whether equivalent associations exclude source senses.
        """
        return self._skip_equivalent

    def for_stage(
        self,
        stage: str,
        *,
        previous_sources: tuple[str, ...] = (),
        skip_equivalent: bool,
    ) -> SynsetCandidates:
        """
        Share the index while selecting one resource or all remaining synsets.

        Args:
            stage: Resource name, or remaining for untried resources.
            previous_sources: Earlier resources excluded from this pass.
            skip_equivalent: Exclude equivalent senses and synsets.

        Returns:
            A lightweight candidate view for the requested pass.
        """
        selected = copy(self)

        selected._stage = stage

        selected._cache_stage = ":".join(
            quote(source, safe="") for source in (*previous_sources, stage)
        )

        selected._previous_sources = previous_sources
        selected._skip_equivalent = skip_equivalent

        return selected

    def _resources(
        self,
        synset: Synset,
    ) -> Iterator[tuple[str, SynsetResource]]:
        """
        Select evidence only from synsets not used in earlier passes.

        Args:
            synset: Candidate whose evidence is requested.

        Yields:
            Resource names and lexical evidence visible in this pass.
        """
        if not self._stage:
            yield from synset.resources.items()

            return

        if any(source in synset.resources for source in self._previous_sources):
            return

        if self._stage == "remaining":
            yield from synset.resources.items()
        elif (resource := synset.resources.get(self._stage)) is not None:
            yield self._stage, resource

    def candidates(
        self,
        lemma: Lemma,
    ) -> tuple[Synset, ...]:
        """
        Retrieve source-eligible synsets for a lemma and its variants.

        Args:
            lemma: Entry whose synsets are requested.

        Returns:
            Unique candidates sorted by synset identifier.
        """
        pos = POS.NOUN if lemma.pos == POS.PROPN else lemma.pos
        matched: dict[str, Synset] = {}

        for form in (lemma.lemma, *sorted(lemma.variants)):
            matched.update(self._members.get((self._normalize(form), pos), {}))

        mapped: set[str] = set()

        if self._skip_equivalent:
            mapped = {
                association.synset_id
                for sense in lemma.senses
                for association in sense.synsets
                if association.relation == SynsetRelation.EQUIVALENT
            }

        forms = {self._normalize(lemma.lemma)} | {
            self._normalize(variant) for variant in lemma.variants
        }

        return tuple(
            synset
            for identifier, synset in sorted(matched.items())
            if identifier not in mapped
            and any(
                self._normalize(member) in forms
                for _, resource in self._resources(synset)
                for member in resource.members
            )
        )

    def synonyms(
        self,
        lemma: Lemma,
        synset: Synset,
    ) -> tuple[str, ...]:
        """
        Return distinct selected members other than the queried lemma.

        Args:
            lemma: Entry whose candidates are being described.
            synset: Candidate lexical concept.

        Returns:
            Lexical members in source order.
        """
        normalized_lemma = self._normalize(lemma.lemma)
        synonyms: dict[str, str] = {}

        for _, resource in self._resources(synset):
            for member in resource.members:
                normalized_member = self._normalize(member)

                if normalized_member != normalized_lemma:
                    _ = synonyms.setdefault(normalized_member, member)

        return tuple(synonyms.values())

    def glosses(
        self,
        synset: Synset,
    ) -> tuple[str, ...]:
        """
        Return distinct glosses from the selected resources.

        Args:
            synset: Candidate lexical concept.

        Returns:
            Glosses in resource order.
        """
        return tuple(
            dict.fromkeys(
                gloss
                for _, resource in self._resources(synset)
                for gloss in resource.glosses
            ),
        )

    def examples(
        self,
        synset: Synset,
    ) -> tuple[str, ...]:
        """
        Return distinct examples from the selected resources.

        Args:
            synset: Candidate lexical concept.

        Returns:
            Examples in resource order.
        """
        return tuple(
            dict.fromkeys(
                example
                for _, resource in self._resources(synset)
                for example in resource.examples
            ),
        )

    def sources(
        self,
        lemma: Lemma,
        synset: Synset,
    ) -> tuple[str, ...]:
        """
        Return selected resources that contain the queried member.

        Args:
            lemma: Entry whose candidate is being described.
            synset: Candidate lexical concept.

        Returns:
            Resource names in input order.
        """
        forms = {self._normalize(lemma.lemma)} | {
            self._normalize(variant) for variant in lemma.variants
        }

        return tuple(
            name
            for name, resource in self._resources(synset)
            if any(self._normalize(member) in forms for member in resource.members)
        )
