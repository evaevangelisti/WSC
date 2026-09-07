"""Exercise manual-reference evaluation and development-only model selection."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from annotation.scripts.alignment import Judgement, consolidate
from experiments.scripts.evaluation import evaluate, summarize
from experiments.scripts.report import analyze_run, build_report, select_run

from wsc.alignment import parse_response, serialize_alignment
from wsc.constants import ALIGNMENT_FIELDS, ALIGNMENT_SCHEMA
from wsc.export import TSVWriter
from wsc.models import POS
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    Definition,
    ModelSettings,
)


def result(
    identifier: str,
    status: str,
) -> AlignmentResult:
    """
    Build a generated WordNet decision.

    Args:
        identifier: Query and headword identifier.
        status: Matched, absent, or uncertain association.

    Returns:
        A validated language model result.
    """
    query = AlignmentQuery(
        AlignmentTask.WORDNET,
        identifier,
        identifier,
        identifier,
        POS.NOUN,
        (Definition("s", ("meaning",)),),
        (Definition("wn", ("concept",)),),
    )
    response = {
        "s": [
            {
                "target_id": "wn",
                "relation": "equivalent",
                "reason": "The definitions identify the same concept.",
            },
        ]
        if status == "matched"
        else None,
    }

    return parse_response(query, json.dumps(response))


def write_run(
    path: Path,
    records: list[AlignmentResult],
    model: str,
) -> None:
    """
    Write a complete experiment cache.

    Args:
        path: Destination table.
        records: Generated decisions.
        model: Configuration label.
    """
    metadata = {
        "schema": ALIGNMENT_SCHEMA,
        "model": model,
        "settings": json.dumps(asdict(ModelSettings(model))),
        "gloss_mode": "last",
        "prompt": "baseline",
        "input": "same-sample",
        "splits": json.dumps(
            {
                record.query.alignment_id: (
                    "development"
                    if record.query.alignment_id.startswith("development")
                    else "test"
                )
                for record in records
            }
        ),
    }

    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata)})

        for record in records:
            for row in serialize_alignment(record):
                writer.write(row)


def test_metrics_count_model_uncertainty_as_abstention() -> None:
    """Model uncertainty retains gold positives in the recall denominator."""
    records = {
        name: result(name, status)
        for name, status in (
            ("a", "matched"),
            ("b", "matched"),
            ("c", "no_match"),
            ("d", "matched"),
        )
    }
    gold = [
        Judgement(
            records["a"].query,
            "1",
            "matched",
            frozenset({("s", "wn", "equivalent")}),
            (),
        ),
        Judgement(
            records["b"].query,
            "1",
            "matched",
            frozenset({("s", "wn", "wiktionary_narrower")}),
            (),
        ),
        Judgement(records["c"].query, "1", "no_match", frozenset(), ()),
        Judgement(records["d"].query, "1", "uncertain", frozenset(), ()),
    ]
    observations = evaluate(records, gold)
    metrics = summarize(observations)

    assert len(observations) == 3
    assert metrics["Precision"] == metrics["Recall"] == metrics["F1"] == 0.5
    assert metrics["No-match accuracy"] == 1
    assert metrics["No-match recall"] == 1
    assert metrics["Exact tasks"] == pytest.approx(2 / 3)


def test_null_decisions_do_not_remove_gold_links() -> None:
    """Abstained positive sources remain false negatives."""
    record = result("test", "no_match")
    gold = Judgement(
        record.query,
        "annotator",
        "matched",
        frozenset({("s", "wn", "equivalent")}),
        (),
    )
    metrics = summarize(evaluate({"test": record}, [gold]))

    assert metrics["Recall"] == 0
    assert metrics["Precision"] is None


def test_report_selects_configurations_using_development_only(
    tmp_path: Path,
) -> None:
    """Held-out changes leave model selection unchanged and notes visible."""
    records = [
        result("development-positive", "matched"),
        result("development-negative", "no_match"),
        result("test-positive", "matched"),
        result("test-negative", "matched"),
    ]
    judgements = [
        Judgement(
            record.query,
            "annotator",
            "matched" if "positive" in record.query.alignment_id else "no_match",
            frozenset({("s", "wn", "equivalent")})
            if "positive" in record.query.alignment_id
            else frozenset(),
            ("Editorial ambiguity | check\ncontext",)
            if record.query.alignment_id == "test-negative"
            else (),
        )
        for record in records
    ]
    gold = consolidate(judgements)
    first_path = tmp_path / "first.tsv"
    second_path = tmp_path / "second.tsv"
    write_run(first_path, records, "first")
    write_run(
        second_path,
        [result(record.query.alignment_id, "matched") for record in records],
        "second",
    )
    first = analyze_run(first_path, gold, 0.99)
    second = analyze_run(second_path, gold, 0.99)
    document = build_report([first_path, second_path], gold, repetitions=5)

    assert select_run([second, first]) is first
    changed = replace(first, observations=())
    assert select_run([second, changed]) is changed
    assert "Threshold" not in document
    assert "precision–recall curve" not in document
    assert "WordNet relations" in document
    assert "Independent agreement before adjudication: unavailable." in document
    assert "Editorial ambiguity \\| check<br>context" in document
    assert "Selected configuration: first / baseline / last / temperature=0" in document
    assert (
        "| Model | Gloss mode | Prompts | Temperature | Reasoning effort |" in document
    )


def test_missing_predictions_are_rejected() -> None:
    """Incomplete model runs cannot silently omit annotated tasks."""
    record = result("missing", "no_match")
    judgement = Judgement(record.query, "1", "no_match", frozenset(), ())

    with pytest.raises(ValueError, match="Prediction context differs"):
        _ = evaluate({}, [judgement])


def test_reports_reject_headword_leakage(
    tmp_path: Path,
) -> None:
    """A shared headword cannot appear in development and test."""
    first = result("development-first", "no_match")
    second = result("test-second", "no_match")
    second = replace(second, query=replace(second.query, lemma=first.query.lemma))
    path = tmp_path / "run.tsv"
    write_run(path, [first, second], "model")
    gold = consolidate(
        [
            Judgement(record.query, "1", "no_match", frozenset(), ())
            for record in (first, second)
        ]
    )

    with pytest.raises(ValueError, match="Headword appears in both"):
        _ = analyze_run(path, gold, 0.99)
