"""
Compare cross-encoders on frozen, model-independent alignment samples.
"""

import argparse
import json
from itertools import product
from pathlib import Path
from typing import TypedDict, cast

from wsc.alignment import CrossEncoderScorer, score_query, serialize_alignment
from wsc.alignment.provenance import build_metadata, cache_key
from wsc.constants import (
    ALIGNMENT_BATCH_SIZE,
    ALIGNMENT_FIELDS,
    ALIGNMENT_MAXIMUM_LENGTH,
    ALIGNMENT_MODELS,
    DEFAULT_INSTRUCTIONS,
    INSTRUCTIONS_PATH,
)
from wsc.export import TSVWriter
from wsc.models.alignment import (
    AlignmentInstructions,
    AlignmentQuery,
    GlossMode,
    Scorer,
)
from wsc.reading import QueryRecord, parse_query, read_instructions, read_metadata
from wsc.upstream import cache


class SampleRecord(TypedDict):
    """Frozen queries and development assignments created before annotation."""

    seed: int
    queries: list[QueryRecord]
    splits: dict[str, str]


def run_experiment(
    queries: list[AlignmentQuery],
    scorer: Scorer,
    mode: GlossMode,
    path: Path,
    metadata: dict[str, str],
    instructions: AlignmentInstructions = DEFAULT_INSTRUCTIONS,
) -> None:
    """
    Persist all candidate scores independently of manual labels.

    Args:
        queries: Frozen shared sample.
        scorer: Semantic cross-encoder.
        mode: Definition representation being evaluated.
        path: Destination TSV cache file.
        metadata: Model, input, and split provenance.
        instructions: Instruction profile being evaluated.
    """
    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps(metadata, ensure_ascii=False)})
        for query in queries:
            for row in serialize_alignment(
                score_query(query, scorer, mode, instructions)
            ):
                writer.write(row)


class Arguments(argparse.Namespace):
    """Typed experiment command options."""

    sample: Path = Path("sample.json")
    model: list[str] | None = None
    gloss_mode: list[GlossMode] | None = None
    revision: str = "main"
    device: str | None = None
    batch_size: int = ALIGNMENT_BATCH_SIZE
    maximum_length: int = ALIGNMENT_MAXIMUM_LENGTH
    cache_dir: Path | None = None
    reuse: bool = False
    instructions: Path = INSTRUCTIONS_PATH
    instruction_profile: list[str] | None = None


def main() -> None:
    """Score the Cartesian product of models, gloss modes, and instruction profiles."""
    parser = argparse.ArgumentParser(description="Score a frozen alignment sample.")
    _ = parser.add_argument("sample", type=Path)
    _ = parser.add_argument(
        "--model", action="append", help="Repeat to select models; defaults to all."
    )
    _ = parser.add_argument(
        "--gloss-mode",
        type=GlossMode,
        choices=GlossMode,
        action="append",
        help="Repeat to select gloss modes; defaults to all.",
    )
    _ = parser.add_argument("--revision", default="main")
    _ = parser.add_argument("--device")
    _ = parser.add_argument("--batch-size", type=int, default=ALIGNMENT_BATCH_SIZE)
    _ = parser.add_argument(
        "--maximum-length", type=int, default=ALIGNMENT_MAXIMUM_LENGTH
    )
    _ = parser.add_argument("--cache-dir", type=Path)
    _ = parser.add_argument("--reuse", action="store_true")
    _ = parser.add_argument("--instructions", type=Path, default=INSTRUCTIONS_PATH)
    _ = parser.add_argument(
        "--instruction-profile",
        action="append",
        help="Repeat to select instruction profiles; defaults to all in the TOML.",
    )
    arguments = parser.parse_args(namespace=Arguments())
    sample = cast(
        SampleRecord, json.loads(arguments.sample.read_text(encoding="utf-8"))
    )
    queries = [parse_query(record) for record in sample["queries"]]
    if not queries or len({query.task for query in queries}) != 1:
        raise ValueError("A sample must contain exactly one alignment task")
    profiles = {
        profile.name: profile for profile in read_instructions(arguments.instructions)
    }
    selected_profiles = [
        profiles[name] for name in arguments.instruction_profile or profiles
    ]
    for model, instructions in product(
        arguments.model or ALIGNMENT_MODELS, selected_profiles
    ):
        scorer = None
        for mode in arguments.gloss_mode or tuple(GlossMode):
            metadata = build_metadata(
                arguments.sample,
                model,
                arguments.revision,
                mode,
                arguments.maximum_length,
                instructions=instructions,
            )
            metadata["splits"] = json.dumps(sample["splits"], sort_keys=True)
            path = (
                cache.alignment_dir(arguments.cache_dir, cache_key(metadata))
                / f"{queries[0].task}.tsv"
            )
            if arguments.reuse:
                if not path.is_file() or read_metadata(path) != metadata:
                    raise ValueError(f"No compatible experiment cache at {path}")
            else:
                if scorer is None:
                    scorer = CrossEncoderScorer(
                        model,
                        revision=arguments.revision,
                        device=arguments.device,
                        batch_size=arguments.batch_size,
                        maximum_length=arguments.maximum_length,
                        instructions=instructions,
                    )
                run_experiment(queries, scorer, mode, path, metadata, instructions)
            print(path)  # noqa: T201
        del scorer


if __name__ == "__main__":
    main()
