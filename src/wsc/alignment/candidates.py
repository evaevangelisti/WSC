"""
Candidate construction independent of model predictions.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator

from ..models import POS, Lemma, Synset
from ..models.alignment import AlignmentQuery, AlignmentTask, Definition


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
        pos = POS.NOUN if lemma.pos == POS.NAME else lemma.pos

        found_synsets: dict[str, Synset] = {}

        variant_forms = (
            variant_id.rsplit(".", 1)[0] for variant_id in sorted(lemma.variants)
        )

        for form in (lemma.lemma, *variant_forms):
            found_synsets.update(self._members.get((self._normalize(form), pos), {}))

        return tuple(found_synsets[identifier] for identifier in sorted(found_synsets))


def build_queries(
    lemma: Lemma,
    task: AlignmentTask,
    candidates: WordNetCandidates,
) -> Iterator[AlignmentQuery]:
    """
    Expose complete candidate sets without semantic filtering.

    Args:
        lemma: Collected entry.
        task: Resource to align.
        candidates: Cached WordNet candidate index.

    Yields:
        One translation entry or one task per WordNet source sense.
    """
    source_definitions = tuple(
        Definition(sense.id, sense.glosses) for sense in lemma.senses
    )

    if task == AlignmentTask.TRANSLATIONS:
        if lemma.translations and source_definitions:
            yield AlignmentQuery(
                task,
                lemma.id,
                lemma.id,
                lemma.lemma,
                lemma.pos,
                source_definitions,
                tuple(
                    Definition(f"translation:{position}", (gloss,))
                    for position, gloss in enumerate(lemma.translations)
                ),
            )

        return

    target_definitions = tuple(
        Definition(synset.id, (synset.definition,))
        for synset in candidates.candidates(lemma)
    )

    for source_definition in source_definitions:
        yield AlignmentQuery(
            task,
            source_definition.id,
            lemma.id,
            lemma.lemma,
            lemma.pos,
            (source_definition,),
            target_definitions,
        )
