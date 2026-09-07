"""Run alignment configuration grids and report manual-reference quality."""

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TypedDict, cast

from annotation.scripts.alignment import consolidate, read_judgements

from wsc.alignment import align_query, serialize_alignment
from wsc.alignment.client import open_model
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import (
    ALIGNMENT_FIELDS,
    ALIGNMENT_MAXIMUM_TOKENS,
    ALIGNMENT_MODEL,
    ALIGNMENT_TEMPERATURE,
    ALIGNMENT_URL,
    DEFAULT_PROMPTS,
    PROMPTS_PATH,
)
from wsc.export import TSVWriter
from wsc.files import partial_file
from wsc.models.alignment import (
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentTask,
    GlossMode,
    LanguageModel,
    ModelSettings,
)
from wsc.reading import QueryRecord, parse_query, read_metadata, read_prompts
from wsc.upstream import cache

from .report import build_report


class SampleRecord(TypedDict):
    """Represent frozen queries and their experimental splits."""

    seed: int
    queries: list[QueryRecord]
    splits: dict[str, str]


@dataclass(frozen=True, slots=True)
class Sample:
    """
    Retain one task's frozen experimental inputs.

    Attributes:
        path: Original sample artifact.
        queries: Shared candidate sets.
        splits: Frozen development and test assignments.
    """

    path: Path
    queries: list[AlignmentQuery]
    splits: dict[str, str]


class Arguments(argparse.Namespace):
    """Define experiment options in alignment command order."""

    samples: Sequence[Path] = ()
    output: Path = Path("experiments/reports")
    task: list[AlignmentTask] | None = None
    model: list[str] | None = None
    url: list[str] | None = None
    gloss_mode: list[GlossMode] | None = None
    prompts: list[Path] | None = None
    temperature: list[float] | None = None
    maximum_tokens: int = ALIGNMENT_MAXIMUM_TOKENS
    reasoning_effort: list[str] | None = None
    reuse: bool = False
    cache_dir: Path | None = None
    annotations: Sequence[Path] = ()
    adjudications: Sequence[Path] = ()
    minimum_precision: float = 0.99
    seed: int = 0
    bootstrap: int = 1000


def read_samples(
    paths: Sequence[Path],
    tasks: list[AlignmentTask] | None,
) -> dict[AlignmentTask, Sample]:
    """
    Read one frozen sample for each requested task.

    Args:
        paths: Sample JSON files produced by annotation preparation.
        tasks: Selected tasks, or all supplied tasks.

    Returns:
        Task-indexed frozen samples.

    Raises:
        ValueError: If samples are empty, duplicated, mixed, or missing.
    """
    samples: dict[AlignmentTask, Sample] = {}

    for path in paths:
        record = cast(SampleRecord, json.loads(path.read_text(encoding="utf-8")))
        queries = [parse_query(item) for item in record["queries"]]
        resources = {query.task for query in queries}

        if len(resources) != 1:
            raise ValueError("A sample must contain exactly one alignment task")

        task = queries[0].task

        if task in samples:
            raise ValueError(f"Multiple samples supplied for {task}")

        samples[task] = Sample(path, queries, record["splits"])

    selected_tasks = tuple(dict.fromkeys(tasks or samples))

    if missing := set(selected_tasks) - samples.keys():
        raise ValueError(f"Missing task samples: {', '.join(sorted(missing))}")

    return {task: samples[task] for task in selected_tasks}


def build_configurations(
    arguments: Arguments,
) -> Iterator[tuple[ModelSettings, GlossMode, AlignmentPrompts]]:
    """
    Enumerate the complete inference configuration grid.

    Args:
        arguments: Requested models, endpoints, and experimental dimensions.

    Yields:
        Model settings, gloss mode, and prompt templates.

    Raises:
        ValueError: If endpoint counts or generation limits are invalid.
    """
    models = arguments.model or [ALIGNMENT_MODEL]
    endpoints = arguments.url or [ALIGNMENT_URL]

    if len(endpoints) == 1:
        endpoints = endpoints * len(models)

    if len(endpoints) != len(models):
        raise ValueError("Supply one endpoint or one endpoint per model")

    temperatures = arguments.temperature or [ALIGNMENT_TEMPERATURE]

    if any(not 0 <= temperature <= 2 for temperature in temperatures):
        raise ValueError("Temperatures must lie between zero and two")

    if arguments.maximum_tokens < 1:
        raise ValueError("Maximum tokens must be positive")

    prompts = [read_prompts(path) for path in arguments.prompts or [PROMPTS_PATH]]

    for endpoint, mode, template, temperature, effort in product(
        zip(models, endpoints, strict=True),
        arguments.gloss_mode or tuple(GlossMode),
        prompts,
        temperatures,
        arguments.reasoning_effort or ["unset"],
    ):
        model, url = endpoint

        settings = ModelSettings(
            model=model,
            temperature=temperature,
            maximum_tokens=arguments.maximum_tokens,
            url=url,
            reasoning_effort=None if effort == "unset" else effort,
        )

        yield settings, mode, template


def run_experiment(
    queries: list[AlignmentQuery],
    model: LanguageModel,
    mode: GlossMode,
    path: Path,
    metadata: dict[str, str],
    prompts: AlignmentPrompts = DEFAULT_PROMPTS,
) -> None:
    """
    Persist model decisions independently of manual labels.

    Args:
        queries: Frozen shared sample.
        model: Language model generation boundary.
        mode: Wiktionary gloss representation.
        path: Destination TSV file.
        metadata: Model settings and experimental splits.
        prompts: Task prompt templates.
    """
    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata, ensure_ascii=False)})

        for query in queries:
            result = align_query(query, model, mode, prompts)

            for row in serialize_alignment(result):
                writer.write(row)


def run_grid(
    sample: Sample,
    arguments: Arguments,
) -> list[Path]:
    """
    Generate or replay every configuration for a frozen task sample.

    Args:
        sample: Task-specific queries and experimental splits.
        arguments: Configuration grid and cache options.

    Returns:
        Completed evidence paths in configuration order.

    Raises:
        ValueError: If requested replay evidence is missing or incompatible.
    """
    paths: list[Path] = []

    for settings, mode, prompts in build_configurations(arguments):
        metadata = build_metadata(sample.path, settings, mode, prompts=prompts)
        metadata["splits"] = json.dumps(sample.splits, sort_keys=True)

        path = (
            cache.alignment_dir(arguments.cache_dir, cache_key(metadata))
            / f"{sample.queries[0].task}.tsv"
        )

        if arguments.reuse:
            if not path.is_file() or read_metadata(path) != metadata:
                raise ValueError(f"No compatible experiment cache at {path}")
        else:
            model = open_model(settings)

            run_experiment(sample.queries, model, mode, path, metadata, prompts)

        paths.append(path)
        print(path)  # noqa: T201

    return paths


def parse_arguments() -> Arguments:
    """
    Parse the experiment grid and manual-reference report options.

    Returns:
        Configuration dimensions, input paths, and reporting options.
    """
    parser = argparse.ArgumentParser(
        description="Run alignment experiments and generate one report per task.",
    )

    _ = parser.add_argument("samples", type=Path, nargs="+")
    _ = parser.add_argument("output", type=Path, help="Report destination directory.")
    _ = parser.add_argument(
        "--task",
        type=AlignmentTask,
        choices=AlignmentTask,
        action="append",
    )
    _ = parser.add_argument("--model", action="append")
    _ = parser.add_argument(
        "--url",
        action="append",
        help="One endpoint or one per model, in model order.",
    )
    _ = parser.add_argument(
        "--gloss-mode",
        type=GlossMode,
        choices=GlossMode,
        action="append",
    )
    _ = parser.add_argument("--prompts", type=Path, action="append")
    _ = parser.add_argument("--temperature", type=float, action="append")
    _ = parser.add_argument(
        "--maximum-tokens",
        type=int,
        default=ALIGNMENT_MAXIMUM_TOKENS,
    )
    _ = parser.add_argument(
        "--reasoning-effort",
        action="append",
        help="Server-supported effort; unset omits the parameter.",
    )
    _ = parser.add_argument("--reuse", action="store_true")
    _ = parser.add_argument("--cache-dir", type=Path)

    _ = parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    _ = parser.add_argument("--adjudications", type=Path, nargs="*", default=[])
    _ = parser.add_argument(
        "--minimum-precision",
        type=float,
        default=Arguments.minimum_precision,
    )
    _ = parser.add_argument("--seed", type=int, default=Arguments.seed)
    _ = parser.add_argument("--bootstrap", type=int, default=Arguments.bootstrap)

    return parser.parse_args(namespace=Arguments())


def main() -> None:
    """Run the task grids and write their quantitative and qualitative reports."""
    arguments = parse_arguments()

    samples = read_samples(arguments.samples, arguments.task)
    judgements = read_judgements(arguments.annotations)
    adjudications = read_judgements(arguments.adjudications)

    references = {
        task: consolidate(
            [item for item in judgements if item.query.task == task],
            [item for item in adjudications if item.query.task == task],
        )
        for task in samples
    }

    if any(not gold.judgements for gold in references.values()):
        raise ValueError("Every selected task requires manual annotations")

    if not 0 <= arguments.minimum_precision <= 1:
        raise ValueError("Minimum precision must lie between zero and one")

    for task, sample in samples.items():
        paths = run_grid(sample, arguments)

        document = build_report(
            paths,
            references[task],
            arguments.minimum_precision,
            arguments.seed,
            arguments.bootstrap,
        )

        output_path = arguments.output / f"{task}.md"

        with partial_file(output_path) as partial:
            _ = partial.write_text(document, encoding="utf-8")

        print(f"Wrote {output_path}")  # noqa: T201


if __name__ == "__main__":
    main()
