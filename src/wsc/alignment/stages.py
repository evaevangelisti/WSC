"""
Plan ordered synset resource passes and their evidence tables.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from ..export.formats.jsonl import JSONLWriter
from ..models import Lemma
from ..models.alignment import (
    AlignmentPrompts,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    LanguageModel,
)
from ..reading import read_lemmas, read_synsets
from .aligner import Aligner
from .batching import TaskCache
from .candidates import SynsetCandidates
from .recording import AlignmentTableKey, alignment_table_key


def select_stages(
    requested: list[str] | None,
    available: frozenset[str],
    tasks: tuple[AlignmentTask, ...],
) -> tuple[str, ...]:
    """
    Validate requested resources and expand the default all-resource pass.

    Args:
        requested: Named resources followed optionally by remaining.
        available: Resource names in the synset input.
        tasks: Alignment tasks selected by the command.

    Returns:
        Ordered pass names, empty for the ordinary all-resource pass.

    Raises:
        ValueError: If a resource is missing, repeated, or out of order.
    """
    if AlignmentTask.SYNSETS not in tasks:
        if requested is not None:
            raise ValueError("--synset-source requires the synsets task")

        return ("",)

    sources = tuple(requested or ("remaining",))

    if len(set(sources)) != len(sources) or "remaining" in sources[:-1]:
        raise ValueError("List each resource once and put remaining last")

    unknown = set(sources) - available - {"remaining"}

    if unknown:
        raise ValueError(f"Unknown synset resources: {', '.join(sorted(unknown))}")

    return ("",) if sources == ("remaining",) else sources


def prepare_synset_stages(
    path: Path,
    requested: list[str] | None,
    tasks: tuple[AlignmentTask, ...],
) -> tuple[SynsetCandidates, tuple[str, ...]]:
    """
    Read synsets once and select the requested resource passes.

    Args:
        path: Synset JSON Lines input.
        requested: Named resources in alignment order.
        tasks: Alignment tasks selected by the command.

    Returns:
        Shared candidate index and ordered pass names.

    Raises:
        ValueError: If the synset input or resource selection is invalid.
    """
    if AlignmentTask.SYNSETS not in tasks:
        return SynsetCandidates(()), select_stages(requested, frozenset(), tasks)

    if not path.is_file():
        raise ValueError(f"No synsets at {path}")

    candidates = SynsetCandidates(read_synsets(path))
    stages = select_stages(requested, candidates.available_sources, tasks)

    return candidates, stages


def alignment_table_paths(
    tasks: tuple[AlignmentTask, ...],
    stages: tuple[str, ...],
    evidence_dir: Path,
) -> dict[AlignmentTableKey, Path]:
    """
    Give each requested task and synset pass a distinct TSV path.

    Args:
        tasks: Alignment tasks selected by the command.
        stages: Ordered synset passes.
        evidence_dir: Directory containing reusable decisions.

    Returns:
        Cache paths indexed by task and optional resource pass.
    """
    paths: dict[AlignmentTableKey, Path] = {
        task: evidence_dir / f"{task}.tsv"
        for task in tasks
        if task != AlignmentTask.SYNSETS
    }

    if AlignmentTask.SYNSETS in tasks:
        for stage in stages:
            name = f"{quote(stage, safe='')}.tsv" if stage else "all.tsv"
            paths[alignment_table_key(AlignmentTask.SYNSETS, stage)] = (
                evidence_dir / "synsets" / name
            )

    return paths


def align_stages(
    entries: Iterable[Lemma],
    candidates: SynsetCandidates,
    tasks: tuple[AlignmentTask, ...],
    stages: tuple[str, ...],
    model: LanguageModel | None,
    mode: GlossMode,
    prompts: AlignmentPrompts,
    batch_size: int,
    cache: Mapping[AlignmentTableKey, TaskCache],
    recorder: Callable[[AlignmentResult], None],
    staging_parent: Path,
    model_loader: Callable[[], LanguageModel] | None = None,
) -> Iterator[Lemma]:
    """
    Finish each resource across all entries before starting the next.

    Args:
        entries: Collected entries read once.
        candidates: Shared synset index.
        tasks: Tasks performed in the first pass.
        stages: Ordered synset resource passes.
        model: Shared model, or None during cached replay.
        mode: Wiktionary gloss representation.
        prompts: Prompt templates for the selected tasks.
        batch_size: Maximum prompts or entries per batch.
        cache: Persisted decisions for every requested pass.
        recorder: Callback writing resolved decisions.
        staging_parent: Parent directory for intermediate aligned entries.
        model_loader: Deferred shared model construction.

    Yields:
        Aligned entries after the final resource pass.
    """
    with TemporaryDirectory(
        prefix=".alignment-stages-",
        dir=staging_parent,
    ) as directory:
        for index, stage in enumerate(stages):
            stage_tasks = tasks if index == 0 else (AlignmentTask.SYNSETS,)

            stage_candidates = (
                candidates
                if not stage
                else candidates.for_stage(
                    stage,
                    previous_sources=stages[:index],
                    skip_equivalent=index > 0,
                )
            )

            stage_cache: dict[AlignmentTask, TaskCache] = {}

            for task in stage_tasks:
                key = alignment_table_key(
                    task,
                    stage if task == AlignmentTask.SYNSETS else "",
                )

                if key in cache:
                    stage_cache[task] = cache[key]

            aligner = Aligner(
                model,
                stage_candidates,
                stage_tasks,
                mode,
                prompts,
                batch_size,
                model_loader,
                prefetch=True,
                show_progress=True,
            )

            aligned = aligner.align(entries, cache=stage_cache, recorder=recorder)

            if index == len(stages) - 1:
                yield from aligned

                continue

            intermediate_path = Path(directory) / f"{index}.jsonl"

            with JSONLWriter[Lemma](intermediate_path) as writer:
                for entry in aligned:
                    writer.write(entry)

            entries = read_lemmas(intermediate_path)
