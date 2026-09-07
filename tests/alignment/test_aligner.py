"""
Public alignment behavior under conflicting, missing, and randomized evidence.
"""

from collections.abc import Sequence
from dataclasses import replace
from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import (
    Aligner,
    WordNetCandidates,
    build_queries,
    score_query,
    select_links,
)
from wsc.models import POS, Example, Lemma, Sense, Synset, WordNetRelation
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Comparison,
    Definition,
    GlossMode,
)


class Scores:
    """Deterministic model substitute leaving assignment and serialization intact."""

    def __init__(self, values: list[float]) -> None:
        """
        Retain scores in query order.

        Args:
            values: Scores supplied by the test.
        """
        self.values: list[float] = values
        self.pairs: list[Comparison] = []

    def score(self, pairs: Sequence[Comparison]) -> list[float]:
        """
        Record model inputs and return the next configured scores.

        Args:
            pairs: Actual semantic hypotheses constructed by the pipeline.

        Returns:
            Configured scores matching the next batch length.
        """
        self.pairs.extend(pairs)
        selected, self.values = self.values[: len(pairs)], self.values[len(pairs) :]

        return selected


@given(
    st.integers(1, 4),
    st.integers(1, 4),
    st.lists(st.integers(-5, 8), min_size=16, max_size=16),
    st.integers(-3, 4),
)
def test_matching_maximizes_partial_assignment(
    rows: int,
    columns: int,
    values: list[int],
    threshold: int,
) -> None:
    """Random matrices match an exhaustive independent assignment oracle."""
    query = AlignmentQuery(
        AlignmentTask.TRANSLATIONS,
        "entry",
        "entry",
        "word",
        POS.NOUN,
        tuple(Definition(str(index), ("sense",)) for index in range(rows)),
        tuple(Definition(str(index), ("gloss",)) for index in range(columns)),
    )
    scores = tuple(
        AlignmentScore(
            str(row), str(column), "translation", float(values[row * columns + column])
        )
        for row in range(rows)
        for column in range(columns)
    )
    selected = select_links(AlignmentResult(query, scores), threshold)
    actual = sum(link.score - threshold for link in selected)
    expected = max(
        sum(
            values[row * columns + column] - threshold
            for row, column in enumerate(assignment)
            if column >= 0
        )
        for assignment in product(range(-1, columns), repeat=rows)
        if len({column for column in assignment if column >= 0})
        == sum(column >= 0 for column in assignment)
    )

    assert actual == expected
    assert len({link.source_id for link in selected}) == len(selected)
    assert len({link.target_id for link in selected}) == len(selected)
    assert all(link.score > threshold for link in selected)


def test_alignment_transfers_only_accepted_translations_and_preserves_input() -> None:
    """Global assignment resolves collisions while preserving source data."""
    senses = [
        Sense("s1", ("first",), sentences=[Example("example")]),
        Sense("s2", ("second",)),
    ]
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=senses,
        translations={
            "group1": {"it": frozenset({"uno"})},
            "group2": {"it": frozenset({"due"})},
            "unrelated": {"it": frozenset({"altro"})},
        },
    )
    aligner = Aligner(
        Scores([10, 9, -5, 8, 0, -5]),
        WordNetCandidates(()),
        {AlignmentTask.TRANSLATIONS: 1},
    )
    aligned = next(aligner.align([lemma]))

    assert aligned.senses[0].translations == {"it": frozenset({"due"})}
    assert aligned.senses[1].translations == {"it": frozenset({"uno"})}
    assert not aligned.translations
    assert len(lemma.translations) == 3
    assert not senses[0].translations
    assert aligned.senses[0].sentences == senses[0].sentences


def test_single_pair_can_abstain_and_still_calls_semantic_model() -> None:
    """A one-by-one entry receives no automatic association."""
    scorer = Scores([-1])
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s", ("meaning",))],
        translations={"meaning": {"it": frozenset({"parola"})}},
    )
    result = next(
        Aligner(scorer, WordNetCandidates(()), {AlignmentTask.TRANSLATIONS: 0}).align(
            [lemma]
        )
    )

    assert len(scorer.pairs) == 1
    assert not result.senses[0].translations
    assert not result.translations


def test_wordnet_preserves_many_to_many_relations() -> None:
    """Senses and synsets retain many-to-many associations."""
    index = WordNetCandidates(
        [
            Synset("wn1", "i1", POS.NOUN, "concept one", ("Word",)),
            Synset("wn2", "i2", POS.NOUN, "concept two", ("Word",)),
        ]
    )
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s1", ("first",)), Sense("s2", ("second",))],
    )
    scorer = Scores([1, 2, 8, 1, 2, 7, 9, 2, 1, -1, -2, -3])
    result = next(Aligner(scorer, index, {AlignmentTask.WORDNET: 5}).align([lemma]))

    assert [(item.synset_id, item.relation) for item in result.senses[0].wordnet] == [
        ("wn1", WordNetRelation.WIKTIONARY_BROADER),
        ("wn2", WordNetRelation.WIKTIONARY_BROADER),
    ]
    assert result.senses[1].wordnet[0].synset_id == "wn1"
    assert result.senses[1].wordnet[0].relation == WordNetRelation.EQUIVALENT


@given(st.text(alphabet="abcABCé .", max_size=15))
def test_candidates_include_variants_and_map_proper_names_to_nouns(
    prefix: str,
) -> None:
    """Variant identifiers preserve dotted spellings and retrieve nominal candidates."""
    spelling = f"{prefix}New_York"
    index = WordNetCandidates(
        [
            Synset("one", "i1", POS.NOUN, "name", (spelling, spelling.lower())),
            Synset("two", "i2", POS.VERB, "verb", (spelling,)),
        ]
    )
    lemma = Lemma(
        "NY.name",
        "NY",
        POS.NAME,
        variants=frozenset({f"{spelling.upper().replace('_', ' ')}.name"}),
        senses=[Sense("s", ("city",))],
    )

    assert [item.ili for item in index.candidates(lemma)] == ["i1"]


@given(st.integers(min_value=1, max_value=15), st.sampled_from(("in", "", "i1")))
def test_synset_identity_preserves_candidates_sharing_an_ili(
    count: int,
    ili: str,
) -> None:
    """Repeated or absent ILI values never merge distinct WordNet candidates."""
    identifiers = {f"wn-{position:02}" for position in range(count)}
    candidates = WordNetCandidates(
        Synset(identifier, ili, POS.NOUN, identifier, ("word",))
        for identifier in sorted(identifiers)
    )
    lemma = Lemma("word.noun", "word", POS.NOUN, senses=[Sense("s", ("sense",))])
    aligner = Aligner(
        Scores([10, 0, 0] * count), candidates, {AlignmentTask.WORDNET: 1}
    )
    aligned = next(aligner.align([lemma]))

    assert {item.id for item in candidates.candidates(lemma)} == identifiers
    assert {item.synset_id for item in aligned.senses[0].wordnet} == identifiers


@pytest.mark.parametrize("mode", list(GlossMode))
def test_representation_changes_only_model_input(mode: GlossMode) -> None:
    """Ablations retain task identities and original hierarchical definitions."""
    query = AlignmentQuery(
        AlignmentTask.TRANSLATIONS,
        "id",
        "lemma",
        "mouse",
        POS.NOUN,
        (Definition("s", ("ancestor marker", "leaf marker")),),
        (Definition("t", ("candidate",)),),
    )
    scorer = Scores([4])
    result = score_query(query, scorer, mode)

    assert result.query == query
    assert "leaf marker" in scorer.pairs[0].query
    assert ("ancestor marker" in scorer.pairs[0].query) == (mode != GlossMode.LAST)


def test_cached_alignment_replays_without_model_and_rejects_changed_definitions() -> (
    None
):
    """Cached replay rejects changed source definitions."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s", ("meaning",))],
        translations={"translation": {"it": frozenset({"parola"})}},
    )
    query = next(
        build_queries(lemma, AlignmentTask.TRANSLATIONS, WordNetCandidates(()))
    )
    result = score_query(query, Scores([5]))
    aligner = Aligner(None, WordNetCandidates(()), {AlignmentTask.TRANSLATIONS: 6})
    aligned = next(
        aligner.align(
            [lemma], cached_results={AlignmentTask.TRANSLATIONS: iter([result])}
        )
    )
    assert not aligned.senses[0].translations
    changed = replace(lemma, senses=[Sense("s", ("different",))])

    with pytest.raises(ValueError, match="Cached candidates differ"):
        _ = list(
            aligner.align(
                [changed], cached_results={AlignmentTask.TRANSLATIONS: iter([result])}
            )
        )


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_invalid_model_evidence_is_rejected(values: list[float]) -> None:
    """Missing and nonfinite model results never reach the matching algorithm."""
    query = AlignmentQuery(
        AlignmentTask.TRANSLATIONS,
        "id",
        "lemma",
        "x",
        POS.NOUN,
        (Definition("s", ("a",)),),
        (Definition("t", ("b",)),),
    )

    with pytest.raises(ValueError, match="finite score"):
        _ = score_query(query, Scores(values))
