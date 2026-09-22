"""Command line for collecting senses out of Wiktionary."""

import json
from contextlib import ExitStack
from functools import partial
from importlib.metadata import version
from logging import DEBUG, getLogger
from pathlib import Path
from typing import Annotated, cast

import typer

from .alignment import (
    Aligner,
    SynsetCandidates,
    open_alignment_recorder,
)
from .alignment.inference import open_model
from .alignment.reporting import AlignmentStatistics, write_alignment
from .alignment.reporting import (
    build_manifest as build_alignment_manifest,
)
from .collection import CollectionSettings, build_manifest, write_collection
from .constants import (
    ALIGNMENT_BATCH_SIZE,
    ALIGNMENT_DIR,
    ALIGNMENT_MAXIMUM_TOKENS,
    ALIGNMENT_MODEL,
    ALIGNMENT_SENSES,
    ALIGNMENT_TEMPERATURE,
    BATCH_SIZE,
    CHUNK_SIZE,
    COLLECTION_DIR,
    KAIKKI_URL,
    LANGUAGE_SECTION,
    PROCESSES,
    PROMPTS_PATH,
    TIMEOUT,
    USER_AGENT,
)
from .extract import (
    DumpExtractor,
    WiktionaryExtractor,
    build_off_page_translations,
    index_translation_glosses,
    narrow,
    open_locator,
    read_entries,
    read_off_page_translations,
    write_off_page_translations,
)
from .logging import configure_logging
from .models import POS, Engine
from .models.alignment import AlignmentResult, AlignmentTask, GlossMode, ModelSettings
from .reading import (
    read_alignment_cache,
    read_lemmas,
    read_prompts,
    read_synsets,
)
from .upstream import cache, download, repository, wiktextract

_LOGGER = getLogger(__name__)

DEFAULT_SYNSETS_PATH = Path("synsets.jsonl")

DumpDate = Annotated[
    str,
    typer.Option(
        envvar="WSC_DUMP_DATE",
        help="Dump to use, as 20260801, or latest.",
    ),
]

CacheDir = Annotated[
    Path | None,
    typer.Option(
        envvar="WSC_CACHE_DIR",
        help="Where the sources and what is made of them are kept.",
        show_default="your platform's cache directory",
    ),
]

app = typer.Typer(
    add_completion=False,
    help="Collect word senses out of Wiktionary.",
)


@app.callback()
def configure(
    context: typer.Context,
) -> None:
    """
    Configure logs for the selected command.

    Args:
        context: Command context that owns the logging handler.
    """
    context.with_resource(configure_logging())


@app.command()
def fetch(
    dump_date: DumpDate = cache.LATEST,
    cache_dir: CacheDir = None,
) -> None:
    """
    Download a Wiktionary dump. Needs the network.
    """
    user_agent = USER_AGENT.format(version=version("wsc"))

    date = dump_date

    if date == cache.LATEST:
        date = repository.latest_date(user_agent, TIMEOUT)
        _LOGGER.info("Resolved latest to %s", date)

    dump_path = cache.dump_dir(cache_dir, date) / cache.DUMP_NAME

    if dump_path.exists():
        _LOGGER.info("Already fetched %s", dump_path)

        return

    download(
        repository.url(date),
        dump_path,
        user_agent,
        TIMEOUT,
        CHUNK_SIZE,
    )

    _LOGGER.info("Fetched %s", dump_path)


@app.command()
def parse(
    dump_date: DumpDate = cache.LATEST,
    processes: Annotated[
        int,
        typer.Option(
            help="Processes wiktextract may run, at 4 GB each.",
        ),
    ] = 1,
    database_path: Annotated[
        Path | None,
        typer.Option(
            "--db-path",
            help="Where the pages extracted from the dump are kept.",
            show_default="a temporary file",
        ),
    ] = None,
    *,
    archive: Annotated[
        bool,
        typer.Option(
            "--archive/--no-archive",
            help="Download the parse kaikki.org publishes instead of making one.",
        ),
    ] = False,
    cache_dir: CacheDir = None,
) -> None:
    """
    Parse a fetched dump with wiktextract, or take one published.

    The dump is then walked for the translations left behind.
    """
    try:
        date = cache.fetched_date(cache_dir, dump_date)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    dump_dir = cache.dump_dir(cache_dir, date)

    dump_path = dump_dir / cache.DUMP_NAME

    if not dump_path.exists():
        raise typer.BadParameter(f"No dump at {dump_path}; fetch it first")

    output_path = dump_dir / cache.WIKTEXTRACT_NAME
    off_page_translations_path = dump_dir / cache.OFF_PAGE_TRANSLATIONS_NAME

    if output_path.exists() and off_page_translations_path.exists():
        _LOGGER.info("Already parsed %s", output_path)

        return

    if not output_path.exists():
        if archive:
            archive_path = dump_dir / cache.ARCHIVE_NAME

            if not archive_path.exists():
                download(
                    KAIKKI_URL,
                    archive_path,
                    USER_AGENT.format(version=version("wsc")),
                    TIMEOUT,
                    CHUNK_SIZE,
                )

            skipped_lines = wiktextract.narrow_file(archive_path, output_path, narrow)
        else:
            skipped_lines = wiktextract.parse(
                dump_path,
                output_path,
                processes,
                narrow,
                database_path,
            )

        if skipped_lines:
            _LOGGER.warning("Skipped %s lines without entries", skipped_lines)

    parsed_glosses = index_translation_glosses(
        read_entries(output_path, "Indexing translation tables"),
    )

    off_page_translations = build_off_page_translations(
        DumpExtractor(LANGUAGE_SECTION).extract(dump_path, parsed_glosses),
        read_entries(output_path, "Answering the pointers"),
    )

    write_off_page_translations(off_page_translations_path, off_page_translations)

    translated = len(off_page_translations)
    _LOGGER.info("Parsed %s: %s off-page translation entries", output_path, translated)


@app.command()
def collect(
    dump_date: DumpDate = cache.LATEST,
    pos: Annotated[
        list[POS] | None,
        typer.Option(
            "--pos",
            help="Parts of speech to keep; repeat to name several.",
            show_default="every part of speech",
        ),
    ] = None,
    minimum_year: Annotated[
        int | None,
        typer.Option(
            "--min-year",
            help="Oldest quotation to keep.",
            show_default="no limit",
        ),
    ] = None,
    maximum_year: Annotated[
        int | None,
        typer.Option(
            "--max-year",
            help="Newest quotation to keep.",
            show_default="no limit",
        ),
    ] = None,
    engine: Annotated[
        Engine,
        typer.Option(
            help="What the sentences are read with to locate the lemma.",
        ),
    ] = Engine.SPACY,
    processes: Annotated[
        int,
        typer.Option(
            help="Processes the reading may run; Stanza runs one regardless.",
        ),
    ] = PROCESSES,
    batch_size: Annotated[
        int,
        typer.Option(
            help="How many sentences the reading takes at a time.",
        ),
    ] = BATCH_SIZE,
    *,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu/--cpu",
            help="Read on the graphics card",
        ),
    ] = False,
    cache_dir: CacheDir = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Directory for the collected senses, reports, and manifest.",
        ),
    ] = COLLECTION_DIR,
) -> None:
    """
    Collect senses, statistics, and provenance into an output directory.
    """
    try:
        date = cache.fetched_date(cache_dir, dump_date)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    dump_dir = cache.dump_dir(cache_dir, date)

    input_path = dump_dir / cache.WIKTEXTRACT_NAME

    if not input_path.exists():
        raise typer.BadParameter(f"Nothing parsed at {input_path}; parse it first")

    off_page_translations_path = dump_dir / cache.OFF_PAGE_TRANSLATIONS_NAME
    off_page_translations = (
        read_off_page_translations(off_page_translations_path)
        if off_page_translations_path.exists()
        else None
    )

    extractor = WiktionaryExtractor(
        frozenset(pos) if pos else None,
        minimum_year,
        maximum_year,
        open_locator(engine, processes, batch_size, gpu=gpu),
        off_page_translations,
    )

    settings = CollectionSettings(
        parts_of_speech=tuple(dict.fromkeys(pos or POS)),
        minimum_year=minimum_year,
        maximum_year=maximum_year,
        engine=engine,
        processes=processes,
        batch_size=batch_size,
        gpu=gpu,
    )

    manifest = build_manifest(
        input_path,
        off_page_translations_path if off_page_translations_path.exists() else None,
        date,
        settings,
    )

    write_collection(extractor.extract(input_path), output_dir, manifest)

    _LOGGER.info("Collected %s", output_dir)


def _parse_option(
    option: str,
) -> tuple[str, object]:
    """
    Parse a single ``key=value`` option.

    Args:
        option: Raw engine or chat template argument.

    Returns:
        The option name and its parsed value.

    Raises:
        BadParameter: If the argument is not in ``key=value`` form.
    """
    name, separator, raw_value = option.partition("=")

    if not separator:
        raise typer.BadParameter(f"Expected key=value, got {option!r}")

    try:
        value = cast(object, json.loads(raw_value))
    except json.JSONDecodeError:
        value = raw_value

    return name, value


@app.command()
def align(
    input_path: Annotated[
        Path,
        typer.Argument(
            metavar="input",
            help="Collected entries to align.",
        ),
    ],
    task: Annotated[
        list[AlignmentTask] | None,
        typer.Option(
            help="Resource to align; repeat to select multiple tasks.",
        ),
    ] = None,
    synsets_path: Annotated[
        Path,
        typer.Option(
            "--synsets",
            help="Generic synsets to align, relative to the current directory.",
        ),
    ] = DEFAULT_SYNSETS_PATH,
    model: Annotated[
        str,
        typer.Option(
            help="Local model path or Hugging Face identifier.",
        ),
    ] = ALIGNMENT_MODEL,
    gloss_mode: Annotated[
        GlossMode,
        typer.Option(
            help="Wiktionary definition representation.",
        ),
    ] = GlossMode.LAST,
    prompts_path: Annotated[
        Path,
        typer.Option(
            "--prompts",
            help="Custom task prompt templates.",
        ),
    ] = PROMPTS_PATH,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=2.0,
            help="Generation temperature.",
        ),
    ] = ALIGNMENT_TEMPERATURE,
    maximum_tokens: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum generated tokens per request.",
        ),
    ] = ALIGNMENT_MAXIMUM_TOKENS,
    batch_size: Annotated[
        int,
        typer.Option(
            min=1,
            help="Prompts prepared for each inference pass.",
        ),
    ] = ALIGNMENT_BATCH_SIZE,
    reasoning_parser: Annotated[
        str | None,
        typer.Option(
            help="vLLM reasoning parser.",
        ),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            help="Reasoning effort supported by the model's chat template.",
        ),
    ] = None,
    engine_option: Annotated[
        list[str] | None,
        typer.Option(
            "--engine-option",
            "-o",
            metavar="KEY=VALUE",
            help="Additional vLLM engine parameter; repeat to set multiple.",
        ),
    ] = None,
    chat_template_option: Annotated[
        list[str] | None,
        typer.Option(
            metavar="KEY=VALUE",
            help="Chat template argument; repeat to set multiple.",
        ),
    ] = None,
    *,
    reuse: Annotated[
        bool,
        typer.Option(
            "--reuse/--recompute",
            help="Reuse cached source decisions and infer only missing senses.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show individual alignment failures and responses.",
        ),
    ] = False,
    cache_dir: CacheDir = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Directory for aligned senses and its reports.",
        ),
    ] = ALIGNMENT_DIR,
) -> None:
    """
    Align collected senses with language model decisions.
    """
    tasks = tuple(dict.fromkeys(task or AlignmentTask))

    if verbose:
        getLogger("wsc").setLevel(DEBUG)

    output_path = output_dir / ALIGNMENT_SENSES

    if input_path.resolve() == output_path.resolve():
        raise typer.BadParameter("Input and output paths must be different")

    if not input_path.is_file():
        raise typer.BadParameter(f"No collection at {input_path}")

    prompts = read_prompts(prompts_path)

    settings = ModelSettings(
        model=model,
        temperature=temperature,
        maximum_tokens=maximum_tokens,
        reasoning_parser=reasoning_parser,
        reasoning_effort=reasoning_effort,
        engine_options=tuple(
            sorted(
                (_parse_option(item) for item in engine_option or ()),
                key=lambda pair: pair[0],
            )
        ),
        chat_template_options=tuple(
            sorted(
                (_parse_option(item) for item in chat_template_option or ()),
                key=lambda pair: pair[0],
            )
        ),
    )

    candidates = SynsetCandidates(())

    if AlignmentTask.SYNSETS in tasks:
        if not synsets_path.is_file():
            raise typer.BadParameter(f"No synsets at {synsets_path}")

        candidates = SynsetCandidates(read_synsets(synsets_path))

    manifest = build_alignment_manifest(
        input_path,
        settings,
        gloss_mode,
        prompts,
        tasks,
    )

    evidence_dir = cache.alignment_dir(cache_dir)
    evidence_paths = {task: evidence_dir / f"{task}.tsv" for task in tasks}

    _LOGGER.info(
        "%s alignment cache: %s",
        "Updating" if reuse else "Writing",
        evidence_dir,
    )

    alignment_cache = {
        selected: read_alignment_cache(path)
        for selected, path in evidence_paths.items()
        if reuse
    }

    language_model = None if reuse else open_model(settings)

    model_loader = partial(open_model, settings) if reuse else None

    statistics = AlignmentStatistics(tasks)

    with ExitStack() as stack:
        recorder = open_alignment_recorder(stack, evidence_paths)

        def record_result(
            result: AlignmentResult,
        ) -> None:
            """
            Persist one result and add it to the task report.

            Args:
                result: Resolved alignment decisions.
            """
            recorder(result)
            statistics.add(result)

        aligner = Aligner(
            language_model,
            candidates,
            tasks,
            gloss_mode,
            prompts,
            batch_size,
            model_loader,
        )

        write_alignment(
            aligner.align(
                read_lemmas(input_path),
                cache=alignment_cache,
                recorder=record_result,
            ),
            output_dir,
            statistics,
            manifest,
        )

    _LOGGER.info("Aligned %s", output_dir)
