"""
Report held-out alignment quality and optional human observations.
"""

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from annotation.scripts.alignment import GoldSample, consolidate, read_judgements

from wsc.files import partial_file
from wsc.models import WordNetRelation
from wsc.models.alignment import AlignmentTask, GlossMode
from wsc.reading import read_alignments, read_metadata

from .evaluation import (
    Observation,
    confidence_intervals,
    evaluate,
    select_threshold,
    summarize,
    threshold_curve,
)


@dataclass(frozen=True, slots=True)
class RunAnalysis:
    """
    Calibrated results for one model, gloss mode, and instruction profile.

    Attributes:
        model: Evaluated cross-encoder identifier.
        gloss_mode: Evaluated source definition representation.
        instruction_profile: Evaluated instruction profile name.
        path: Original cached evidence.
        threshold: Development-selected abstention boundary.
        reached: Whether a nonempty development point met the precision target.
        development: Number of determinate development tasks.
        observations: Held-out manual comparisons.
        curve: Development-only precision and recall curve.
    """

    model: str
    gloss_mode: GlossMode
    instruction_profile: str
    path: Path
    threshold: float
    reached: bool
    development: int
    observations: tuple[Observation, ...]
    curve: tuple[tuple[float, dict[str, float | None]], ...]

    @property
    def name(self) -> str:
        """Describe all three configuration dimensions."""
        return f"{self.model} / {self.instruction_profile} / {self.gloss_mode}"


def analyze_run(path: Path, gold: GoldSample, minimum_precision: float) -> RunAnalysis:
    """
    Tune on frozen development tasks and evaluate their held-out counterparts.

    Args:
        path: Experiment TSV generated from the frozen annotation sample.
        gold: Consolidated manually annotated tasks.
        minimum_precision: Empirical development precision target.

    Returns:
        Model analysis without using test labels for threshold selection.

    Raises:
        ValueError: If splits, resource types, or annotated task contexts differ.
    """
    metadata = read_metadata(path)
    splits = cast(dict[str, str], json.loads(metadata["splits"]))
    records = list(read_alignments(path))
    results = {result.query.alignment_id: result for result in records}
    if len(results) != len(records) or set(splits) != set(results):
        raise ValueError("Experiment must contain each frozen task exactly once")
    if len({result.query.task for result in records}) != 1:
        raise ValueError("An experiment must contain one alignment resource")
    if set(splits.values()) - {"development", "test"}:
        raise ValueError("Unknown frozen experiment split")
    grouped: dict[str, str] = {}
    for identifier, result in results.items():
        lemma = result.query.lemma.casefold()
        if lemma in grouped and grouped[lemma] != splits[identifier]:
            raise ValueError(f"Headword appears in both development and test: {lemma}")
        grouped[lemma] = splits[identifier]
    _ = evaluate(results, gold.judgements, 0.0)
    development = [
        item
        for item in gold.judgements
        if splits[item.query.alignment_id] == "development"
    ]
    test = [
        item for item in gold.judgements if splits[item.query.alignment_id] == "test"
    ]
    curve = threshold_curve(results, development)
    threshold, reached = select_threshold(curve, minimum_precision)
    observations = evaluate(results, test, threshold)
    if not observations:
        raise ValueError("Evaluation requires determinate held-out manual annotations")

    return RunAnalysis(
        metadata["model"],
        GlossMode(metadata["gloss_mode"]),
        metadata["instruction_profile"],
        path,
        threshold,
        reached,
        sum(item.status in {"matched", "no_match"} for item in development),
        tuple(observations),
        tuple(curve),
    )


def select_run(runs: Sequence[RunAnalysis]) -> RunAnalysis | None:
    """
    Choose a configuration using development metrics alone.

    Args:
        runs: Configurations with independently calibrated development thresholds.

    Returns:
        Highest-recall feasible configuration, or None if precision is unattainable.
    """
    eligible = [run for run in runs if run.reached]
    if not eligible:
        return None

    def rank(run: RunAnalysis) -> tuple[float, float]:
        """
        Rank the selected development operating point.

        Args:
            run: Configuration whose threshold was selected on development data.

        Returns:
            Recall followed by precision for deterministic tie-breaking.
        """
        metrics = dict(run.curve)[run.threshold]

        return metrics["Recall"] or 0.0, metrics["Precision"] or 0.0

    return max(eligible, key=rank)


def _format(value: float | None) -> str:
    """
    Render undefined metrics distinctly from measured zero.

    Args:
        value: Optional metric value.

    Returns:
        Percentage or an explicit undefined marker.
    """
    return "undefined" if value is None else f"{value:.1%}"


def _escape(text: str) -> str:
    """
    Keep annotation text inside its Markdown table cell.

    Args:
        text: Optional annotator prose or task identifier.

    Returns:
        Escaped cell text preserving line breaks.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


def _write_comparisons(
    runs: Sequence[RunAnalysis],
    split: Literal["development", "test"],
) -> list[str]:
    """
    Compare each model, gloss mode, and instruction combination separately.

    Args:
        runs: Configurations calibrated independently on development annotations.
        split: Annotation split whose metrics are displayed.

    Returns:
        One row per configuration with an exactly reproducible score threshold.
    """
    heading = (
        "Development configuration search"
        if split == "development"
        else "Held-out configuration comparison"
    )
    lines = [
        f"## {heading}",
        "",
        "Feasibility always refers to the development precision target.",
        "",
        (
            "| Model | Gloss mode | Instructions | Threshold | Precision | Recall | "
            "F1 | Source coverage | Feasible |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        metrics = (
            dict(run.curve)[run.threshold]
            if split == "development"
            else summarize(run.observations)
        )
        cells = (
            _escape(run.model),
            run.gloss_mode,
            _escape(run.instruction_profile),
            repr(run.threshold),
            *(
                _format(metrics[metric])
                for metric in ("Precision", "Recall", "F1", "Source coverage")
            ),
            str(run.reached),
        )
        lines.append(f"| {' | '.join(cells)} |")

    return [*lines, ""]


def _write_run(run: RunAnalysis, seed: int, repetitions: int) -> list[str]:
    """
    Render held-out metrics, strata, and the development operating curve.

    Args:
        run: One evaluated model representation.
        seed: Headword bootstrap seed.
        repetitions: Number of bootstrap replicates.

    Returns:
        Markdown lines for the model analysis.
    """
    metrics = summarize(run.observations)
    intervals = confidence_intervals(run.observations, seed, repetitions)
    lines = [
        f"## {_escape(run.name)}",
        "",
        f"Evidence: `{run.path}`. Threshold: `{run.threshold!r}`.",
        "",
        (
            f"Development tasks: {run.development}. "
            f"Test tasks: {len(run.observations)}. "
            f"Development precision target reached: {run.reached}."
        ),
        "",
        "| Metric | Test value | 95% headword bootstrap interval |",
        "| --- | ---: | --- |",
    ]
    for metric, value in metrics.items():
        interval = intervals[metric]
        bounds = (
            "undefined"
            if interval is None
            else f"{_format(interval[0])}–{_format(interval[1])}"
        )
        lines.append(f"| {metric} | {_format(value)} | {bounds} |")
    lines += [
        "",
        "### Test strata",
        "",
        "| Stratum | Tasks | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    strata = {
        **{
            f"POS: {pos}": [
                item for item in run.observations if item.gold.query.pos == pos
            ]
            for pos in sorted({item.gold.query.pos for item in run.observations})
        },
        "Nested definitions": [
            item
            for item in run.observations
            if any(
                len(source.glosses) > 1 for source in item.gold.query.source_definitions
            )
        ],
        "Top-level definitions": [
            item
            for item in run.observations
            if all(
                len(source.glosses) == 1
                for source in item.gold.query.source_definitions
            )
        ],
        "Multiple candidates": [
            item
            for item in run.observations
            if len(item.gold.query.target_definitions) > 1
        ],
    }
    for name, items in strata.items():
        measured = summarize(items)
        values = " | ".join(
            _format(measured[key]) for key in ("Precision", "Recall", "F1")
        )
        lines.append(f"| {name} | {len(items)} | {values} |")
    if run.observations[0].gold.query.task == AlignmentTask.WORDNET:
        lines += _write_relations(run.observations)
    lines += [
        "",
        "### Development precision–recall curve",
        "",
        "| Threshold | Precision | Recall | Source coverage |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for threshold, measured in run.curve:
        values = " | ".join(
            _format(measured[key]) for key in ("Precision", "Recall", "Source coverage")
        )
        lines.append(f"| {threshold!r} | {values} |")

    return lines


def _write_relations(observations: Sequence[Observation]) -> list[str]:
    """
    Report exact directed relation correctness on held-out WordNet tasks.

    Args:
        observations: WordNet test comparisons.

    Returns:
        Relation-specific precision, recall, and F1 rows.
    """
    lines = [
        "",
        "### WordNet relations",
        "",
        "| Relation | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for relation in WordNetRelation:
        subset = [
            Observation(
                replace(
                    item.gold,
                    links=frozenset(
                        link for link in item.gold.links if link[2] == relation
                    ),
                ),
                frozenset(link for link in item.predicted if link[2] == relation),
            )
            for item in observations
        ]
        measured = summarize(subset)
        values = " | ".join(
            _format(measured[key]) for key in ("Precision", "Recall", "F1")
        )
        lines.append(f"| {relation} | {values} |")

    return lines


def build_report(
    paths: Sequence[Path],
    gold: GoldSample,
    minimum_precision: float = 0.99,
    seed: int = 0,
    repetitions: int = 1000,
) -> str:
    """
    Compare models using shared manual references and optional qualitative notes.

    Args:
        paths: Cached experiments on the same frozen sample.
        gold: Consolidated annotations and adjudication status.
        minimum_precision: Empirical development precision target.
        seed: Reproducible bootstrap seed.
        repetitions: Number of headword bootstrap replicates.

    Returns:
        Quantitative and qualitative Markdown report.

    Raises:
        ValueError: If model samples or frozen splits differ.
    """
    identities = [
        (read_metadata(path)["input"], read_metadata(path)["splits"]) for path in paths
    ]
    if len(set(identities)) != 1:
        raise ValueError(
            "Model comparison requires identical samples and frozen splits"
        )
    runs = [analyze_run(path, gold, minimum_precision) for path in paths]
    statuses = Counter(item.status for item in gold.judgements)
    lines = [
        "# Alignment experiments",
        "",
        (
            "Manual annotations are the only reference. "
            "Thresholds and configuration selection use development labels. "
            "Development and held-out comparisons are reported separately."
        ),
        "",
        (
            f"Unique resolved annotations: {len(gold.judgements)}. "
            f"Uncertain: {statuses['uncertain']}. "
            f"Unresolved disagreements: {len(gold.conflicts)}."
        ),
        "",
        f"Independent agreement before adjudication: {gold.agreed}/{gold.repeated}.",
        "",
        (
            "Link metrics exclude uncertain and unresolved tasks. "
            "Recall is conditional on candidates, not end-to-end WordNet recall."
        ),
        "",
        (
            "This is a pilot sample. Empirical precision targets and degenerate "
            "bootstrap intervals do not establish population precision."
        ),
        "",
    ]
    winner = select_run(runs)
    lines += _write_comparisons(runs, "development")
    lines += [
        "",
        f"Selected configuration: {_escape(winner.name)}."
        if winner is not None
        else "No configuration meets the development precision target.",
        "",
        "Selection maximizes recall, then precision; ties keep input order.",
        "Test comparisons do not select thresholds or the winning configuration.",
        "Further tuning after inspecting test outcomes requires a new test set.",
        "",
    ]
    lines += _write_comparisons(runs, "test")
    selected_runs = [winner] if winner is not None else []
    for run in selected_runs:
        lines.extend(_write_run(run, seed, repetitions))
        lines.append("")
    lines += [
        "## Annotator notes",
        "",
        "| Task | Status | Configuration outcomes | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for item in gold.judgements:
        if not item.notes:
            continue
        outcomes: list[str] = []
        for run in runs:
            observations = {
                observation.gold.query.alignment_id: observation
                for observation in run.observations
            }
            observation = observations.get(item.query.alignment_id)
            outcome = (
                "excluded or development"
                if observation is None
                else (
                    "exact"
                    if observation.predicted == item.links
                    else "incorrect or incomplete"
                )
            )
            outcomes.append(f"{run.name}: {outcome}")
        cells = (
            _escape(item.query.alignment_id),
            item.status,
            _escape("; ".join(outcomes)),
            _escape("\n".join(item.notes)),
        )
        lines.append(f"| {' | '.join(cells)} |")
    if gold.conflicts:
        lines += ["", "Unresolved tasks: " + ", ".join(map(_escape, gold.conflicts))]

    return "\n".join((*lines, ""))


class Arguments(argparse.Namespace):
    """Typed report command options."""

    runs: Sequence[Path] = ()
    annotations: Sequence[Path] = ()
    adjudications: Sequence[Path] = ()
    output: Path = Path("experiments/reports/alignment.md")
    minimum_precision: float = 0.99
    seed: int = 0
    bootstrap: int = 1000


def main() -> None:
    """Generate a manual-reference report from cached experiment TSV files."""
    parser = argparse.ArgumentParser(
        description="Evaluate alignments against manual annotations."
    )
    _ = parser.add_argument("runs", type=Path, nargs="+")
    _ = parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    _ = parser.add_argument("--adjudications", type=Path, nargs="*", default=[])
    _ = parser.add_argument("--output", type=Path, default=Arguments.output)
    _ = parser.add_argument(
        "--minimum-precision", type=float, default=Arguments.minimum_precision
    )
    _ = parser.add_argument("--seed", type=int, default=Arguments.seed)
    _ = parser.add_argument("--bootstrap", type=int, default=Arguments.bootstrap)
    arguments = parser.parse_args(namespace=Arguments())
    gold = consolidate(
        read_judgements(arguments.annotations), read_judgements(arguments.adjudications)
    )
    document = build_report(
        arguments.runs,
        gold,
        arguments.minimum_precision,
        arguments.seed,
        arguments.bootstrap,
    )
    with partial_file(arguments.output) as partial:
        _ = partial.write_text(document, encoding="utf-8")
    print(f"Wrote {arguments.output}")  # noqa: T201


if __name__ == "__main__":
    main()
