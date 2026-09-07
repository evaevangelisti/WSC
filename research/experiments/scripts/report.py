"""Report held-out alignment quality and optional human observations."""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from annotation.scripts.alignment import GoldSample

from wsc.models import WordNetRelation
from wsc.models.alignment import AlignmentTask, GlossMode
from wsc.reading import read_alignments, read_metadata

from .evaluation import (
    Observation,
    confidence_intervals,
    evaluate,
    summarize,
)


@dataclass(frozen=True, slots=True)
class RunAnalysis:
    """
    Store evaluation results for one model, gloss mode, and prompt template.

    Attributes:
        model: Evaluated language model identifier.
        gloss_mode: Evaluated source definition representation.
        prompt_name: Evaluated prompt file name.
        temperature: Requested sampling temperature.
        reasoning_effort: Requested reasoning effort, or the server default.
        url: Model endpoint used for inference.
        path: Original cached evidence.
        reached: Whether development predictions met the precision target.
        development: Number of determinate development tasks.
        observations: Held-out manual comparisons.
        development_observations: Comparisons on development annotations.
    """

    model: str
    gloss_mode: GlossMode
    prompt_name: str
    temperature: float
    reasoning_effort: str | None
    url: str
    path: Path
    reached: bool
    development: int
    observations: tuple[Observation, ...]
    development_observations: tuple[Observation, ...]

    @property
    def name(
        self,
    ) -> str:
        """Identify the model, representation, prompts, and generation settings."""
        return (
            f"{self.model} / {self.prompt_name} / {self.gloss_mode} / "
            f"temperature={self.temperature:g} / "
            f"reasoning={self.reasoning_effort or 'unset'}"
        )


def analyze_run(
    path: Path,
    gold: GoldSample,
    minimum_precision: float,
) -> RunAnalysis:
    """
    Evaluate frozen development and held-out tasks separately.

    Args:
        path: Experiment TSV generated from the frozen annotation sample.
        gold: Consolidated manually annotated tasks.
        minimum_precision: Empirical development precision target.

    Returns:
        Separate development and held-out comparisons.

    Raises:
        ValueError: If splits, resource types, or annotated task contexts differ.
    """
    metadata = read_metadata(path)
    settings = cast(dict[str, object], json.loads(metadata["settings"]))
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

    _ = evaluate(results, gold.judgements)

    development = [
        item
        for item in gold.judgements
        if splits[item.query.alignment_id] == "development"
    ]
    test = [
        item for item in gold.judgements if splits[item.query.alignment_id] == "test"
    ]

    development_observations = evaluate(results, development)
    precision = summarize(development_observations)["Precision"]
    reached = precision is not None and precision >= minimum_precision

    observations = evaluate(results, test)

    if not observations:
        raise ValueError("Evaluation requires determinate held-out manual annotations")

    return RunAnalysis(
        metadata["model"],
        GlossMode(metadata["gloss_mode"]),
        metadata["prompt"],
        cast(float, settings["temperature"]),
        cast(str | None, settings["reasoning_effort"]),
        cast(str, settings["url"]),
        path,
        reached,
        sum(item.status in {"matched", "no_match"} for item in development),
        tuple(observations),
        tuple(development_observations),
    )


def select_run(
    runs: Sequence[RunAnalysis],
) -> RunAnalysis | None:
    """
    Choose a configuration using development metrics alone.

    Args:
        runs: Configurations evaluated on shared development annotations.

    Returns:
        Highest-recall feasible configuration, or None if precision is unattainable.
    """
    eligible = [run for run in runs if run.reached]

    if not eligible:
        return None

    def rank(
        run: RunAnalysis,
    ) -> tuple[float, float]:
        """
        Rank a configuration using development metrics.

        Args:
            run: Evaluated configuration.

        Returns:
            Recall followed by precision for deterministic tie-breaking.
        """
        metrics = summarize(run.development_observations)

        return metrics["Recall"] or 0.0, metrics["Precision"] or 0.0

    return max(eligible, key=rank)


def _format(
    value: float | None,
) -> str:
    """
    Render undefined metrics distinctly from measured zero.

    Args:
        value: Optional metric value.

    Returns:
        Percentage or an explicit undefined marker.
    """
    return "undefined" if value is None else f"{value:.1%}"


def _escape(
    text: str,
) -> str:
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
    Compare each model, gloss mode, and prompt combination separately.

    Args:
        runs: Configurations evaluated on shared manual annotations.
        split: Annotation split whose metrics are displayed.

    Returns:
        One row per configuration with metrics for the requested split.
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
            "| Model | Gloss mode | Prompts | Temperature | Reasoning effort | "
            "Precision | Recall | "
            "F1 | Source coverage | Feasible |"
        ),
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for run in runs:
        metrics = (
            summarize(run.development_observations)
            if split == "development"
            else summarize(run.observations)
        )
        cells = (
            _escape(run.model),
            run.gloss_mode,
            _escape(run.prompt_name),
            f"{run.temperature:g}",
            _escape(run.reasoning_effort or "unset"),
            *(
                _format(metrics[metric])
                for metric in ("Precision", "Recall", "F1", "Source coverage")
            ),
            str(run.reached),
        )
        lines.append(f"| {' | '.join(cells)} |")

    return [*lines, ""]


def _write_run(
    run: RunAnalysis,
    seed: int,
    repetitions: int,
) -> list[str]:
    """
    Render held-out metrics, strata, and relation comparisons.

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
        f"Evidence: `{run.path}`.",
        f"Endpoint: {_escape(run.url)}.",
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

    return lines


def _write_relations(
    observations: Sequence[Observation],
) -> list[str]:
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

    if not 0 <= minimum_precision <= 1:
        raise ValueError("Minimum precision must lie between zero and one")

    runs = [analyze_run(path, gold, minimum_precision) for path in paths]
    statuses = Counter(item.status for item in gold.judgements)
    lines = [
        "# Alignment experiments",
        "",
        (
            "Manual annotations are the only reference. "
            "Configuration selection uses development labels. "
            "Development and held-out comparisons are reported separately."
        ),
        "",
        (
            f"Unique resolved annotations: {len(gold.judgements)}. "
            f"Uncertain: {statuses['uncertain']}. "
            f"Unresolved disagreements: {len(gold.conflicts)}."
        ),
        "",
        (
            f"Independent agreement before adjudication: {gold.agreed}/{gold.repeated}."
            if gold.repeated
            else "Independent agreement before adjudication: unavailable."
        ),
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
        "Test comparisons do not select the winning configuration.",
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
