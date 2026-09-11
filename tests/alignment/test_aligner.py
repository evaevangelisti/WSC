"""Exercise generated decisions through the public alignment API."""

import json
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import (
    Aligner,
    WordNetCandidates,
    align_query,
    build_queries,
    build_request,
    parse_response,
)
from wsc.extract.identifiers import translation_table_id
from wsc.models import (
    POS,
    Lemma,
    Sense,
    Synset,
    TranslationTable,
    WordNetAlignment,
    WordNetRelation,
)
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentTask,
    Definition,
    GlossMode,
    ModelRequest,
)


class Model:
    """Return configured responses and retain the generated requests."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        """
        Store generated responses.

        Args:
            responses: Model responses in request order.
        """
        self.responses: list[str] = list(responses)
        self.requests: list[ModelRequest] = []

    def generate(
        self,
        request: ModelRequest,
    ) -> str:
        """
        Record a request and return its response.

        Args:
            request: Actual rendered model request.

        Returns:
            The next configured response.
        """
        self.requests.append(request)

        return self.responses.pop(0)


def query(
    task: AlignmentTask = AlignmentTask.TRANSLATIONS,
) -> AlignmentQuery:
    """
    Build a two-source query with overlapping contextual vocabulary.

    Args:
        task: Resource being aligned.

    Returns:
        Complete source and candidate definitions.
    """
    return AlignmentQuery(
        task,
        "word.noun",
        "word.noun",
        "word",
        POS.NOUN,
        (
            Definition("s1", ("parent", "first sense"), ("synonym",)),
            Definition("s2", ("second sense",)),
        ),
        (
            Definition("t1", ("first heading",), ("target synonym",)),
            Definition("t2", ("second heading",)),
        ),
    )


def decision(
    target: str | None,
    relation: str = "translation",
) -> list[dict[str, str]] | None:
    """
    Build a generated association or abstention.

    Args:
        target: Accepted candidate identifier.
        relation: Directed semantic relation.

    Returns:
        A supported association or null.
    """
    if target is None:
        return None

    return [
        {
            "target_id": target,
            "relation": relation,
            "reason": "The definitions express the same concept.",
        },
    ]


@given(targets=st.permutations(("t1", "t2")), abstain=st.booleans())
def test_translation_decisions_preserve_one_to_one_associations(
    targets: tuple[str, ...],
    *,
    abstain: bool,
) -> None:
    """Generated assignments retain explicit omissions and distinct targets."""
    response = json.dumps(
        {"s1": decision(targets[0]), "s2": decision(None if abstain else targets[1])}
    )
    result = align_query(query(), Model([response]))

    assert result.response == response
    assert result.decisions[0].links[0].target_id == targets[0]
    assert len(result.links) == (1 if abstain else 2)
    assert len({link.target_id for link in result.links}) == len(result.links)


@pytest.mark.parametrize(
    "response",
    [
        {"s1": decision("t1")},
        {"s1": decision("t1"), "s2": decision("t1")},
        {"s1": decision("unknown"), "s2": decision(None)},
        {"s1": decision("t1", "equivalent"), "s2": decision(None)},
        {"s1": {"status": "matched", "links": []}, "s2": decision(None)},
        {
            "s1": {
                "status": "uncertain",
                "links": [{"target_id": "t1", "relation": "translation"}],
            },
            "s2": decision(None),
        },
    ],
)
def test_invalid_model_assignments_are_rejected(
    response: dict[str, object],
) -> None:
    """Missing, contradictory, and invented associations fail validation."""
    with pytest.raises(ValueError, match=r"Expected|Invalid|One-to-one"):
        _ = align_query(query(), Model([json.dumps(response)]))


@pytest.mark.parametrize("response", ["null", "[]", "```json\n{}\n```", "{"])
def test_invalid_model_json_is_rejected(
    response: str,
) -> None:
    """Malformed model responses remain visible failures."""
    with pytest.raises(ValueError, match=r"Expected|Expecting"):
        _ = parse_response(query(), response)


def test_empty_candidates_avoid_model_inference() -> None:
    """Empty candidate sets produce explicit empty decisions."""
    model = Model([])
    empty = align_query(replace(query(), target_definitions=()), model)

    assert all(not item.links for item in empty.decisions)
    assert len(empty.decisions) == len(query().source_definitions)
    assert not model.requests


def test_alignment_applies_decisions_and_preserves_collection() -> None:
    """Translations move to copied senses while source entries remain intact."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s1", ("first",)), Sense("s2", ("second",))],
        translation_tables=(
            TranslationTable(
                translation_table_id("word.noun", "first heading"),
                "first heading",
                {"it": frozenset({"uno"})},
            ),
            TranslationTable(
                translation_table_id("word.noun", "second heading"),
                "second heading",
                {"it": frozenset({"due"})},
            ),
        ),
    )
    model = Model(
        [
            json.dumps(
                {
                    "s1": decision(translation_table_id("word.noun", "second heading")),
                    "s2": decision(translation_table_id("word.noun", "first heading")),
                }
            )
        ]
    )
    aligner = Aligner(model, WordNetCandidates(()), (AlignmentTask.TRANSLATIONS,))
    (aligned,) = aligner.align([lemma])

    assert aligned.senses[0].translations == {"it": frozenset({"due"})}
    assert aligned.senses[1].translations == {"it": frozenset({"uno"})}
    assert not aligned.translation_tables
    assert lemma.translation_tables
    assert all(not sense.translations for sense in lemma.senses)


def test_invalid_model_response_skips_the_lemma() -> None:
    """A malformed generation does not stop the remaining alignment stream."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s", ("sense",))],
        translation_tables=(
            TranslationTable(
                translation_table_id("word.noun", "heading"),
                "heading",
                {"it": frozenset({"uno"})},
            ),
        ),
    )
    model = Model(["not JSON"])
    aligner = Aligner(model, WordNetCandidates(()), (AlignmentTask.TRANSLATIONS,))

    (aligned,) = tuple(aligner.align([lemma]))

    assert aligned.id == lemma.id
    assert not aligned.translation_tables


def test_wordnet_keeps_multiple_synsets_and_directed_relations() -> None:
    """A source retains equivalent and broader WordNet candidates."""
    lemma = Lemma("word.noun", "word", POS.NOUN, senses=[Sense("s", ("sense",))])
    synsets = (
        Synset("wn1", "i1", POS.NOUN, "specific", ("word", "synonym")),
        Synset("wn2", "i2", POS.NOUN, "general", ("word",)),
    )
    response = json.dumps(
        {
            "s": [
                {
                    "target_id": "wn1",
                    "relation": "equivalent",
                    "reason": "Same concept.",
                },
                {
                    "target_id": "wn2",
                    "relation": "wiktionary_narrower",
                    "reason": "The source adds a defining restriction.",
                },
            ],
        }
    )
    model = Model([response])
    aligner = Aligner(model, WordNetCandidates(synsets), (AlignmentTask.WORDNET,))
    (aligned,) = aligner.align([lemma])

    assert aligned.senses[0].wordnet == (
        WordNetAlignment("wn1", WordNetRelation.EQUIVALENT),
        WordNetAlignment("wn2", WordNetRelation.WIKTIONARY_NARROWER),
    )
    assert "wn1 (synonym) specific" in model.requests[0].prompt
    assert not lemma.senses[0].wordnet


@pytest.mark.parametrize("mode", list(GlossMode))
def test_prompts_render_identifiers_synonyms_and_hierarchy(
    mode: GlossMode,
) -> None:
    """Prompt definitions retain their identity and selected context."""
    prompt = build_request(query(), mode).prompt

    assert (
        "s1 (synonym) parent > first sense"
        if mode == GlossMode.FULL
        else "s1 (synonym) first sense"
    ) in prompt
    assert "t1 (target synonym) first heading" in prompt
    assert "s2 second sense" in prompt
    assert ("Read each" in prompt) == (mode == GlossMode.FULL)
    assert '"status"' not in prompt
    assert tuple(GlossMode) == (GlossMode.LAST, GlossMode.FULL)


def test_candidates_include_variants_and_sense_synonyms() -> None:
    """Variant lookup preserves dotted forms and sense-specific synonyms."""
    lemma = Lemma(
        "alias.name",
        "alias",
        POS.PROPN,
        variants=frozenset({"a.b.noun"}),
        senses=[Sense("s", ("sense",), synonyms=("variant",))],
    )
    candidates = WordNetCandidates(
        [Synset("wn", "ili", POS.NOUN, "definition", ("a.b",))]
    )
    (result,) = build_queries(lemma, AlignmentTask.WORDNET, candidates)

    assert result.source_definitions[0].synonyms == ("variant",)
    assert result.target_definitions[0].id == "wn"
    assert result.target_definitions[0].synonyms == ("a.b",)


def test_candidates_exclude_the_queried_lemma_from_synonyms() -> None:
    """Candidate context does not repeat the lemma as a WordNet synonym."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s", ("sense",))],
    )
    candidates = WordNetCandidates(
        [Synset("wn", "ili", POS.NOUN, "definition", ("word", "term", "word_form"))]
    )

    (result,) = build_queries(lemma, AlignmentTask.WORDNET, candidates)

    assert result.target_definitions[0].synonyms == ("term", "word_form")


def test_replay_rejects_context_drift_and_extra_results() -> None:
    """Cached decisions must cover the current collection exactly."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s1", ("first",))],
        translation_tables=(
            TranslationTable(
                translation_table_id("word.noun", "heading"),
                "heading",
                {"it": frozenset({"uno"})},
            ),
        ),
    )
    candidates = WordNetCandidates(())
    (sample,) = build_queries(lemma, AlignmentTask.TRANSLATIONS, candidates)
    result = parse_response(
        sample,
        json.dumps(
            {"s1": decision(translation_table_id("word.noun", "heading"))}
        ),
    )
    aligner = Aligner(None, candidates, (AlignmentTask.TRANSLATIONS,))

    with pytest.raises(ValueError, match="Unused cached"):
        _ = list(
            aligner.align(
                [lemma],
                cached_results={AlignmentTask.TRANSLATIONS: iter([result, result])},
            )
        )

    changed = replace(lemma, senses=[Sense("s1", ("changed",))])

    with pytest.raises(ValueError, match="Cached candidates differ"):
        _ = list(
            aligner.align(
                [changed], cached_results={AlignmentTask.TRANSLATIONS: iter([result])}
            )
        )
