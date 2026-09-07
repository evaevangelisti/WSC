"""Build shared annotation samples for offsets and semantic alignments."""

import argparse
from pathlib import Path

from corpus import DATA

from wsc.alignment import WordNetCandidates, build_queries
from wsc.models.alignment import AlignmentTask
from wsc.reading import read_lemmas, read_synsets
from wsc.upstream import cache

from .alignment import write_tasks
from .offsets import build_offsets
from .sampling import COUNT, HERE, SEED, sample_items


class Arguments(argparse.Namespace):
    """Typed annotation command options."""

    task: str = "offsets"
    input: Path = DATA / "spacy.jsonl"
    data_dir: Path = DATA
    output: Path | None = None
    count: int = COUNT
    seed: int = SEED
    wordnet_edition: str = cache.LATEST
    cache_dir: Path | None = None


def main() -> None:
    """Build task-specific primary samples and independent second assignments."""
    parser = argparse.ArgumentParser(description="Build shared annotation samples.")
    _ = parser.add_argument(
        "task", choices=("offsets", *AlignmentTask), nargs="?", default="offsets"
    )
    _ = parser.add_argument("--input", type=Path, default=DATA / "spacy.jsonl")
    _ = parser.add_argument("--data-dir", type=Path, default=DATA)
    _ = parser.add_argument("--output", type=Path)
    _ = parser.add_argument("--count", type=int, default=COUNT)
    _ = parser.add_argument("--seed", type=int, default=SEED)
    _ = parser.add_argument("--wordnet-edition", default=cache.LATEST)
    _ = parser.add_argument("--cache-dir", type=Path)
    arguments = parser.parse_args(namespace=Arguments())
    output = arguments.output or HERE / "tasks" / arguments.task

    if arguments.task == "offsets":
        build_offsets(arguments.data_dir, output, arguments.count, arguments.seed)
    else:
        task = AlignmentTask(arguments.task)
        index = WordNetCandidates(())

        if task == AlignmentTask.WORDNET:
            synsets_path = cache.fetched_edition(
                arguments.cache_dir, arguments.wordnet_edition
            )

            if synsets_path is None:
                parser.error("WordNet is not cached; run 'wsc wordnet' first.")

            index = WordNetCandidates(read_synsets(synsets_path))

        queries = (
            query
            for lemma in read_lemmas(arguments.input)
            for query in build_queries(lemma, task, index)
            if len(index.candidates(lemma)) != 0
        )
        write_tasks(
            sample_items(queries, arguments.count, arguments.seed),
            output,
            arguments.seed,
        )

    print(f"Wrote {arguments.count} shared tasks and second assignments to {output}")  # noqa: T201


if __name__ == "__main__":
    main()
