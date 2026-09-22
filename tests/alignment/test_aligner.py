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
    AlignmentResult,
    AlignmentTask,
    GlossMode,
)

from .examples import Model, build_decision, build_query


@given(targets=st.permutations(("t1", "t2")), abstain=st.booleans())
def test_preserves_translation_bijection(
    targets: tuple[str, ...],
    *,
    abstain: bool,
) -> None:
    """Generated assignments retain explicit omissions and distinct targets."""
    response = json.dumps(
        {
            "s1": build_decision(targets[0]),
            "s2": build_decision(None if abstain else targets[1]),
        },
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
        {"s1": [], "s2": None},
        {"s1": ["t1"], "s2": None},
        {"s1": [{"target_id": "t1", "relation": "translation"}], "s2": None},
        {
            "s1": [
                {"target_id": "t1", "relation": "translation", "reason": "Same."},
            ]
            * 2,
            "s2": None,
        },
    ],
)
def test_rejects_invalid_assignments(
    response: dict[str, object],
) -> None:
    """Missing, contradictory, and invented associations fail validation."""
    with pytest.raises(ValueError, match=r"Expected|Invalid|One-to-one|Repeated"):
        _ = align_query(build_query(), Model([json.dumps(response)]))


@pytest.mark.parametrize("reason", [None, 0, False, [], {}, "", " \n\t"])
def test_rejects_invalid_reasons(
    reason: object,
) -> None:
    """Associations require textual evidence containing more than whitespace."""
    response = json.dumps(
        {
            "s1": [{"target_id": "t1", "relation": "translation", "reason": reason}],
            "s2": None,
        },
    )

    with pytest.raises(ValueError, match="Invalid association"):
        _ = align_query(build_query(), Model([response]))


@pytest.mark.parametrize("response", ["null", "[]", "```json\n{}\n```", "{"])
def test_rejects_invalid_json(
    response: str,
) -> None:
    """Malformed model responses remain visible failures."""
    with pytest.raises(ValueError, match=r"Expected|Expecting"):
        _ = parse_response(build_query(), response)


def test_skips_empty_candidates() -> None:
    """Empty candidate sets produce explicit empty decisions."""
    model = Model([])
    empty = align_query(replace(build_query(), target_definitions=()), model)

    assert all(not item.links for item in empty.decisions)
    assert len(empty.decisions) == len(build_query().source_definitions)
    assert not model.requests


def test_aligns_collection_copies() -> None:
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
                        translation_table_id("word.noun", "second heading"),
                    ),
                    "s2": build_decision(
                        translation_table_id("word.noun", "first heading"),
                    ),
                },
            ),
        ],
    )
    aligner = Aligner(model, WordNetCandidates(()), (AlignmentTask.TRANSLATIONS,))
    (aligned,) = aligner.align([lemma])

    assert aligned.senses[0].translation_table == lemma.translation_tables[1]
    assert aligned.senses[1].translation_table == lemma.translation_tables[0]
    assert not aligned.translation_tables
    assert lemma.translation_tables
    assert all(sense.translation_table is None for sense in lemma.senses)


def test_preserves_directed_relations() -> None:
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
        },
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


@given(
    relations=st.tuples(*(st.sampled_from((None, *WordNetRelation)) for _ in range(4))),
)
@example(relations=(WordNetRelation.EQUIVALENT, WordNetRelation.EQUIVALENT, None, None))
@example(relations=(WordNetRelation.EQUIVALENT, None, WordNetRelation.EQUIVALENT, None))
@example(
    relations=(
        WordNetRelation.EQUIVALENT,
        WordNetRelation.WIKTIONARY_NARROWER,
        WordNetRelation.WIKTIONARY_BROADER,
        WordNetRelation.EQUIVALENT,
    ),
)
@example(
    relations=(
        WordNetRelation.WIKTIONARY_NARROWER,
        WordNetRelation.WIKTIONARY_BROADER,
        WordNetRelation.WIKTIONARY_BROADER,
        WordNetRelation.WIKTIONARY_NARROWER,
    ),
)
def test_requires_unique_equivalences(
    relations: tuple[WordNetRelation | None, ...],
) -> None:
    """Only equivalence requires distinct sources and targets in generated graphs."""
    pairs = (("s1", "t1"), ("s1", "t2"), ("s2", "t1"), ("s2", "t2"))
    associations = [
        (source, target, relation)
        for (source, target), relation in zip(pairs, relations, strict=True)
        if relation is not None
    ]
    equivalents = [
        (source, target)
        for source, target, relation in associations
        if relation == WordNetRelation.EQUIVALENT
    ]

    response = json.dumps(
        {
            source_id: [
                {
                    "target_id": target,
                    "relation": relation,
                    "reason": "The definitions support the association.",
                }
                for source, target, relation in associations
                if source == source_id
            ]
            or None
            for source_id in ("s1", "s2")
        },
    )
    query = build_query(AlignmentTask.WORDNET)
    model = Model([response])

    if len({source for source, _ in equivalents}) != len(equivalents) or len(
        {target for _, target in equivalents},
    ) != len(equivalents):
        with pytest.raises(ValueError, match="One-to-one alignment violated"):
            _ = align_query(query, model)
    else:
        result = align_query(query, model)

        assert [
            (link.source_id, link.target_id, link.relation) for link in result.links
        ] == associations


@pytest.mark.parametrize("mode", list(GlossMode))
def test_renders_definition_context(
    mode: GlossMode,
) -> None:
    """Prompt definitions retain their identity and selected context."""
    prompt = build_request(build_query(), mode).prompt

    expected_gloss = "parent > first sense" if mode == GlossMode.FULL else "first sense"

    assert f'"synonyms":["synonym"],"gloss":"{expected_gloss}"' in prompt
    assert '{"id":"t1","gloss":"first heading"}' in prompt
    assert '{"id":"s2","gloss":"second sense"}' in prompt
    assert ("Read each" in prompt) == (mode == GlossMode.FULL)
    assert '"status"' not in prompt


def test_includes_variant_candidates() -> None:
    """Variant lookup preserves dotted forms and sense-specific synonyms."""
    lemma = Lemma(
        "alias.name",
        "alias",
        POS.PROPN,
        variants=frozenset({"a.b"}),
        senses=[Sense("s", ("sense",), synonyms=("variant",))],
    )
    candidates = WordNetCandidates(
        [Synset("wn", "ili", POS.NOUN, "definition", ("a.b",))],
    )
    (result,) = build_queries(lemma, AlignmentTask.WORDNET, candidates)

    assert result.source_definitions[0].synonyms == ("variant",)
    assert len(result.source_definitions) == 1
    assert result.target_definitions[0].id == "wn"
    assert result.target_definitions[0].synonyms == ("a.b",)
    assert result.alignment_id == "wordnet:alias.name"


def test_excludes_headword_synonyms() -> None:
    """Candidate context does not repeat the lemma as a WordNet synonym."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense("s", ("sense",))],
    )
    candidates = WordNetCandidates(
        [Synset("wn", "ili", POS.NOUN, "definition", ("word", "term", "word_form"))],
    )

    (result,) = build_queries(lemma, AlignmentTask.WORDNET, candidates)

    assert len(result.source_definitions) == 1
    assert result.alignment_id == "wordnet:word.noun"
    assert result.target_definitions[0].synonyms == ("term", "word_form")


def test_queries_complete_senses() -> None:
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


@given(
    decisions=st.lists(st.tuples(st.booleans(), st.booleans()), max_size=12),
    targets=st.data(),
)
def test_reuses_available_source_decisions(
    decisions: list[tuple[bool, bool]],
    targets: st.DataObject,
) -> None:
    """Any cached subset preserves decisions, order, and deferred model loading."""
    assigned = targets.draw(st.permutations(tuple(range(len(decisions)))))
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        senses=[Sense(f"s{index}", (f"meaning {index}",)) for index in assigned],
        translation_tables=tuple(
            TranslationTable(
                f"t{index}", f"heading {index}", {"it": frozenset({str(index)})}
            )
            for index in assigned
        ),
    )
    candidates = WordNetCandidates(())
    queries = tuple(build_queries(lemma, AlignmentTask.TRANSLATIONS, candidates))
    recorded: list[AlignmentResult] = []
    responses = {
        f"s{index}": build_decision(f"t{assigned[index]}" if matched else None)
        for index, (_, matched) in enumerate(decisions)
    }
    cached = {
        source: value
        for index, (source, value) in enumerate(responses.items())
        if decisions[index][0]
    }
    pending = {
        source: value for source, value in responses.items() if source not in cached
    }
    model = Model([json.dumps(pending)] if pending else [])
    loads: list[None] = []

    def load_model() -> Model:
        """Record deferred construction when at least one source needs inference."""
        loads.append(None)

        return model

    cache = {}

    if queries:
        query = queries[0]
        cached_query = replace(
            query,
            source_definitions=tuple(
                source for source in query.source_definitions if source.id in cached
            ),
        )
        cache = {
            AlignmentTask.TRANSLATIONS: {
                query.alignment_id: {
                    decision.source_id: decision
                    for decision in parse_response(
                        cached_query, json.dumps(cached)
                    ).decisions
                },
            },
        }

    (aligned,) = Aligner(
        None,
        candidates,
        (AlignmentTask.TRANSLATIONS,),
        model_loader=load_model,
    ).align([lemma], cache=cache, recorder=recorded.append)

    assert len(loads) == bool(pending)
    assert len(model.requests) == bool(pending)
    assert [sense.id for sense in aligned.senses] == [
        sense.id for sense in lemma.senses
    ]
    assert all(sense.translation_table is None for sense in lemma.senses)
    assert aligned.translation_tables == ()

    for sense in aligned.senses:
        index = int(sense.id[1:])
        expected_table = next(
            table
            for table in lemma.translation_tables
            if table.id == f"t{assigned[index]}"
        )

        assert sense.translation_table == (
            expected_table if decisions[index][1] else None
        )

    if pending:
        assert model.requests[0].schema["required"] == [
            sense.id for sense in lemma.senses if sense.id in pending
        ]

    if queries:
        assert len(recorded) == 1
        assert (
            recorded[0].decisions
            == parse_response(queries[0], json.dumps(responses)).decisions
        )
    else:
        assert not recorded


@given(failures=st.lists(st.booleans(), min_size=1, max_size=15))
@example(failures=[True, False, True])
def test_isolates_failed_queries(
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
                    f"s{index}",
                    ("meaning",),
                    translation_table=TranslationTable(
                        f"old{index}",
                        "old meaning",
                        {"it": frozenset({"old"})},
                    ),
                ),
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

    aligned = list(Aligner(model, candidates, batch_size=4).align(lemmas))

    assert len(model.requests) == len(responses)
    assert model.batches == [
        min(4, len(responses) - start) for start in range(0, len(responses), 4)
    ]
    assert len(aligned) == len(lemmas)
    assert len(caplog.records) == sum(failures)

    for original, result, failed in zip(lemmas, aligned, failures, strict=True):
        assert result.translation_tables == (
            original.translation_tables if failed else ()
        )
        assert result.senses[0].translation_table == (
            original.senses[0].translation_table
            if failed
            else original.translation_tables[0]
        )
        assert result.senses[0].wordnet == (
            WordNetAlignment("wn", WordNetRelation.EQUIVALENT),
        )
        assert original.translation_tables
        assert original.senses[0].translation_table is not None
        assert not original.senses[0].wordnet
