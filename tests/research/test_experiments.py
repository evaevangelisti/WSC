"""
Manual-reference metrics, threshold isolation, and qualitative reporting.
"""

import json
from dataclasses import replace
from math import nextafter
from pathlib import Path

import pytest
from annotation.scripts.alignment import Judgement, consolidate
from experiments.scripts.evaluation import evaluate, summarize
from experiments.scripts.report import (
    RunAnalysis,
    analyze_run,
    build_report,
    select_run,
)
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import serialize_alignment
from wsc.constants import ALIGNMENT_FIELDS
from wsc.export import TSVWriter
from wsc.models import POS
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Definition,
    GlossMode,
)


def result(identifier: str, score: float) -> AlignmentResult:
    """
    Provide one WordNet candidate with all relation scores.

    Args:
        identifier: Task and headword identifier.
        score: Equivalent-relation evidence.

    Returns:
        Complete directed relation evidence.
    """
    query = AlignmentQuery(
        AlignmentTask.WORDNET,
        identifier,
        identifier,
        identifier,
        POS.NOUN,
        (Definition("s", ("meaning",)),),
        (Definition("i1", ("concept",)),),
    )

    return AlignmentResult(
        query,
        (
            AlignmentScore("s", "i1", "equivalent", score),
            AlignmentScore("s", "i1", "wiktionary_narrower", -10),
            AlignmentScore("s", "i1", "wiktionary_broader", -10),
        ),
    )


def test_metrics_distinguish_wrong_relations_abstention_and_uncertainty() -> None:
    """Relation errors reduce precision and recall; uncertainty is excluded."""
    records = {
        name: result(name, score)
        for name, score in (("a", 5), ("b", 4), ("c", -2), ("d", 7))
    }
    gold = [
        Judgement(
            records["a"].query,
            "1",
            "matched",
            frozenset({("s", "i1", "equivalent")}),
            (),
        ),
        Judgement(
            records["b"].query,
            "1",
            "matched",
            frozenset({("s", "i1", "wiktionary_narrower")}),
            (),
        ),
        Judgement(records["c"].query, "1", "no_match", frozenset(), ()),
        Judgement(records["d"].query, "1", "uncertain", frozenset(), ()),
    ]
    observations = evaluate(records, gold, 0)
    metrics = summarize(observations)

    assert len(observations) == 3
    assert metrics["Precision"] == metrics["Recall"] == metrics["F1"] == 0.5
    assert metrics["Source coverage"] == pytest.approx(2 / 3)
    assert metrics["No-match accuracy"] == metrics["No-match recall"] == 1
    assert metrics["Exact tasks"] == pytest.approx(2 / 3)


def test_report_tunes_only_development_and_includes_manual_notes(
    tmp_path: Path,
) -> None:
    """Test-score changes cannot influence the selected development threshold."""
    records = [
        result("development-positive", 5),
        result("development-negative", 1),
        result("test-positive", 4),
        result("test-negative", 2),
    ]
    judgements = [
        Judgement(
            record.query,
            "1",
            "matched" if "positive" in record.query.alignment_id else "no_match",
            frozenset({("s", "i1", "equivalent")})
            if "positive" in record.query.alignment_id
            else frozenset(),
            ("Editorial ambiguity | check\ncontext",)
            if record.query.alignment_id == "test-negative"
            else (),
        )
        for record in records
    ]
    metadata = {
        "model": "fake",
        "revision": "fixed",
        "gloss_mode": "last",
        "instruction_profile": "baseline",
        "input": "same",
        "splits": json.dumps(
            {
                record.query.alignment_id: "development"
                if record.query.alignment_id.startswith("development")
                else "test"
                for record in records
            }
        ),
    }
    path = tmp_path / "wordnet.tsv"
    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata, ensure_ascii=False)})
        for record in records:
            for row in serialize_alignment(record):
                writer.write(row)
    gold = consolidate(judgements)
    original = analyze_run(path, gold, 0.99)
    document = build_report([path], gold, repetitions=10)
    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata, ensure_ascii=False)})
        for record in (
            *records[:2],
            result("test-positive", -100),
            result("test-negative", 100),
        ):
            for row in serialize_alignment(record):
                writer.write(row)
    changed = analyze_run(path, gold, 0.99)

    assert original.threshold == changed.threshold == 1
    assert "Editorial ambiguity \\| check<br>context" in document
    assert "WordNet relations" in document
    assert "Development precision–recall curve" in document
    assert "95% headword bootstrap interval" in document


def test_missing_prediction_is_not_silently_removed() -> None:
    """Incomplete model runs cannot improve metrics through selective omissions."""
    judgement = Judgement(result("missing", 1).query, "1", "no_match", frozenset(), ())

    with pytest.raises(ValueError, match="Prediction context differs"):
        _ = evaluate({}, [judgement], 0)


def test_configuration_selection_uses_development_metrics_only(tmp_path: Path) -> None:
    """Changing held-out outcomes never changes the selected configuration."""
    first = RunAnalysis(
        "first",
        GlossMode.LAST,
        "baseline",
        tmp_path / "first.tsv",
        1.0,
        reached=True,
        development=20,
        observations=(),
        curve=((1.0, {"Recall": 0.8, "Precision": 1.0}),),
    )
    second = replace(
        first,
        model="second",
        curve=((1.0, {"Recall": 0.5, "Precision": 1.0}),),
    )
    record = result("held-out", 5)
    gold = Judgement(record.query, "1", "no_match", frozenset(), ())
    incorrect = evaluate({record.query.alignment_id: record}, [gold], 0)
    changed = replace(first, observations=tuple(incorrect))

    assert select_run([second, first]) is first
    assert select_run([second, changed]) is changed
    assert select_run([replace(first, reached=False)]) is None


@given(st.integers(min_value=0, max_value=100))
def test_report_compares_all_factors_without_selecting_test_winners(
    tmp_path: Path,
    boundary: int,
) -> None:
    """Reports preserve thresholds without selecting held-out winners."""
    threshold = nextafter(float(boundary), float("inf"))
    records = [
        result("development-positive", threshold + 1),
        result("development-negative", threshold),
        result("test-positive", threshold - 1),
        result("test-negative", threshold + 2),
    ]
    judgements = [
        Judgement(
            record.query,
            "annotator",
            "matched" if "positive" in record.query.alignment_id else "no_match",
            frozenset({("s", "i1", "equivalent")})
            if "positive" in record.query.alignment_id
            else frozenset(),
            ("Manual explanation",)
            if record.query.alignment_id == "test-positive"
            else (),
        )
        for record in records
    ]
    alternative = [
        result("development-positive", threshold - 1),
        result("development-negative", threshold + 1),
        result("test-positive", threshold + 2),
        result("test-negative", threshold - 1),
    ]
    paths: list[Path] = []
    for model, mode, profile, results in (
        ("model-a", "last", "baseline", records),
        ("model-b", "context", "entailment", alternative),
    ):
        path = tmp_path / f"{model}.tsv"
        paths.append(path)
        metadata = {
            "model": model,
            "gloss_mode": mode,
            "instruction_profile": profile,
            "input": "shared",
            "splits": json.dumps(
                {
                    record.query.alignment_id: "development"
                    if record.query.alignment_id.startswith("development")
                    else "test"
                    for record in records
                }
            ),
        }
        with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
            writer.write({"context": json.dumps(metadata)})
            for record in results:
                for row in serialize_alignment(record):
                    writer.write(row)
    document = build_report(paths, consolidate(judgements), repetitions=2)
    development, held_out = document.split("## Held-out configuration comparison")
    for section in (development, held_out):
        assert "| Model | Gloss mode | Instructions | Threshold |" in section
        assert f"| model-a | last | baseline | {threshold!r} |" in section
        assert "| model-b | context | entailment |" in section
    rows = [
        line.strip("| ").split(" | ")
        for line in held_out.splitlines()
        if line.startswith(("| model-a |", "| model-b |"))
    ]

    assert float(rows[0][3]) == threshold
    assert rows[0][4:7] == ["0.0%", "0.0%", "0.0%"]
    assert rows[1][4:7] == ["100.0%", "100.0%", "100.0%"]
    assert rows[1][-1] == "False"
    assert "Selected configuration: model-a / baseline / last." in document
    assert "model-a / baseline / last: incorrect or incomplete" in document
    assert "model-b / entailment / context: exact" in document
