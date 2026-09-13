"""Exercise generated decisions through the public alignment API."""

import json
from dataclasses import replace

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from wsc.alignment import (
    Aligner,
    WordNetCandidates,
    align_query,
    build_queries,
    build_request,
    parse_response,
)
from wsc.identifiers import translation_table_id
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
    AlignmentTask,
    GlossMode,
)

from .examples import Model, build_decision, build_query


@given(targets=st.permutations(("t1", "t2")), abstain=st.booleans())
def test_translation_decisions_preserve_one_to_one_associations(
    targets: tuple[str, ...],
    *,
    abstain: bool,
) -> None:
    """Generated assignments retain explicit omissions and distinct targets."""
    response = json.dumps(
        {
            "s1": build_decision(targets[0]),
            "s2": build_decision(None if abstain else targets[1]),
        }
    )
    result = align_query(build_query(), Model([response]))

    assert result.response == response
    assert result.decisions[0].links[0].target_id == targets[0]
    assert len(result.links) == (1 if abstain else 2)
    assert len({link.target_id for link in result.links}) == len(result.links)


@pytest.mark.parametrize(
    "response",
    [
        {"s1": build_decision("t1")},
        {"s1": build_decision("t1"), "s2": build_decision("t1")},
        {"s1": build_decision("unknown"), "s2": build_decision(None)},
        {"s1": build_decision("t1", "equivalent"), "s2": build_decision(None)},
        {"s1": {"status": "matched", "links": []}, "s2": build_decision(None)},
        {
            "s1": {
                "status": "uncertain",
                "links": [{"target_id": "t1", "relation": "translation"}],
            },
            "s2": build_decision(None),
        },
    ],
)
def test_invalid_model_assignments_are_rejected(
    response: dict[str, object],
) -> None:
    """Missing, contradictory, and invented associations fail validation."""
    with pytest.raises(ValueError, match=r"Expected|Invalid|One-to-one"):
        _ = align_query(build_query(), Model([json.dumps(response)]))


@pytest.mark.parametrize("response", ["null", "[]", "```json\n{}\n```", "{"])
def test_invalid_model_json_is_rejected(
    response: str,
) -> None:
    """Malformed model responses remain visible failures."""
    with pytest.raises(ValueError, match=r"Expected|Expecting"):
        _ = parse_response(build_query(), response)


def test_empty_candidates_avoid_model_inference() -> None:
    """Empty candidate sets produce explicit empty decisions."""
    model = Model([])
    empty = align_query(replace(build_query(), target_definitions=()), model)

    assert all(not item.links for item in empty.decisions)
    assert len(empty.decisions) == len(build_query().source_definitions)
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
                    "s1": build_decision(
                        translation_table_id("word.noun", "second heading")
                    ),
                    "s2": build_decision(
                        translation_table_id("word.noun", "first heading")
                    ),
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
    prompt = build_request(build_query(), mode).prompt

    assert (
        "s1 (synonym) parent > first sense"
        if mode == GlossMode.FULL
        else "s1 (synonym) first sense"
    ) in prompt
    assert "t1 (target synonym) first heading" in prompt
    assert "s2 second sense" in prompt
    assert ("Read each" in prompt) == (mode == GlossMode.FULL)
    assert '"status"' not in prompt


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
    assert len(result.source_definitions) == 1
    assert result.target_definitions[0].id == "wn"
    assert result.target_definitions[0].synonyms == ("a.b",)
    assert result.alignment_id == "wordnet:alias.name"


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

    assert len(result.source_definitions) == 1
    assert result.alignment_id == "wordnet:word.noun"
    assert result.target_definitions[0].synonyms == ("term", "word_form")


def test_wordnet_query_contains_all_source_senses() -> None:
    """WordNet compares every Wiktionary sense in one model request."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[
            Sense("s1", ("first sense",)),
            Sense("s2", ("second sense",)),
        ],
    )

    (result,) = build_queries(
        lemma,
        AlignmentTask.WORDNET,
        WordNetCandidates([Synset("wn", "ili", POS.NOUN, "definition", ("word",))]),
    )

    assert tuple(source.id for source in result.source_definitions) == ("s1", "s2")
    assert result.alignment_id == "wordnet:word.noun"


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
    assert sample.alignment_id == "translations:word.noun"
    result = parse_response(
        sample,
        json.dumps(
            {"s1": build_decision(translation_table_id("word.noun", "heading"))}
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


@given(failures=st.lists(st.booleans(), min_size=1, max_size=15))
@example(failures=[True, False, True])
def test_failed_queries_preserve_translations_and_allow_other_tasks(
    failures: list[bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failures preserve source data while later resources and entries still align."""
    lemmas = [
        Lemma(
            f"word{index}.noun",
            "word",
            POS.NOUN,
            senses=[
                Sense(
                    f"s{index}", ("meaning",), translations={"it": frozenset({"old"})}
                )
            ],
            translation_tables=(
                TranslationTable(f"t{index}", "meaning", {"it": frozenset({"new"})}),
            ),
        )
        for index in range(len(failures))
    ]
    responses = [
        response
        for index, failed in enumerate(failures)
        for response in (
            "invalid"
            if failed
            else json.dumps({f"s{index}": build_decision(f"t{index}")}),
            json.dumps({f"s{index}": build_decision("wn", "equivalent")}),
        )
    ]
    model = Model(responses)
    candidates = WordNetCandidates([Synset("wn", "i1", POS.NOUN, "meaning", ("word",))])
    caplog.clear()
    aligned = list(Aligner(model, candidates).align(lemmas))

    assert len(model.requests) == len(responses)
    assert len(aligned) == len(lemmas)
    assert len(caplog.records) == sum(failures)

    for original, result, failed in zip(lemmas, aligned, failures, strict=True):
        assert result.translation_tables == (
            original.translation_tables if failed else ()
        )
        assert result.senses[0].translations == {
            "it": frozenset({"old" if failed else "new"})
        }
        assert result.senses[0].wordnet == (
            WordNetAlignment("wn", WordNetRelation.EQUIVALENT),
        )
        assert original.translation_tables
        assert original.senses[0].translations == {"it": frozenset({"old"})}
        assert not original.senses[0].wordnet
