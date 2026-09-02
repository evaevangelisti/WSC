"""
Command line for collecting senses out of Wiktionary.
"""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from .constants import BATCH_SIZE, CHUNK_SIZE, PROCESSES, TIMEOUT, USER_AGENT
from .export import Writer, open_writer
from .extract import WiktionaryExtractor, open_locator
from .models import POS, Engine, Lemma
from .upstream import cache, download, repository, wiktextract

# Named apart from the signatures, so that the commands asking for the same
# option share one.

Language = Annotated[
    str,
    typer.Option(
        envvar="WSC_LANGUAGE",
        help="Wiktionary edition to read, by language code.",
    ),
]

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


@app.command()
def fetch(
    language: Language = "en",
    dump_date: DumpDate = cache.LATEST,
    cache_dir: CacheDir = None,
) -> None:
    """
    Download a Wiktionary dump. Needs the network.
    """
    user_agent = USER_AGENT.format(version=version("wsc"))

    date = dump_date
    if date == cache.LATEST:
        date = repository.latest_date(language, user_agent, TIMEOUT)
        typer.echo(f"Resolved latest to {date}")

    dump_path = cache.dump_dir(cache_dir, language, date) / cache.DUMP_NAME
    if dump_path.exists():
        typer.echo(f"Already fetched {dump_path}")
        return

    download(
        repository.url(language, date),
        dump_path,
        user_agent,
        TIMEOUT,
        CHUNK_SIZE,
    )

    typer.echo(f"Fetched {dump_path}")


@app.command()
def parse(
    language: Language = "en",
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
    cache_dir: CacheDir = None,
) -> None:
    """
    Parse a fetched dump with wiktextract.
    """
    try:
        date = cache.fetched_date(cache_dir, language, dump_date)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    dump_dir = cache.dump_dir(cache_dir, language, date)

    output_path = dump_dir / cache.WIKTEXTRACT_NAME
    if output_path.exists():
        typer.echo(f"Already parsed {output_path}")
        return

    dump_path = dump_dir / cache.DUMP_NAME
    if not dump_path.exists():
        raise typer.BadParameter(f"No dump at {dump_path}; fetch it first")

    skipped_lines = wiktextract.parse(
        dump_path,
        output_path,
        language,
        processes,
        database_path,
    )

    if skipped_lines:
        typer.echo(f"Set aside {skipped_lines} lines of wiktextract's own reporting")

    typer.echo(f"Parsed {output_path}")


@app.command()
def collect(
    output_path: Annotated[
        Path,
        typer.Argument(
            metavar="output",
            help="Where the senses go; the suffix picks the format.",
        ),
    ],
    language: Language = "en",
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
) -> None:
    """
    Collect the senses of a parsed dump into a file.
    """
    try:
        date = cache.fetched_date(cache_dir, language, dump_date)
    except FileNotFoundError as error:
        raise typer.BadParameter(str(error)) from error

    input_path = cache.dump_dir(cache_dir, language, date) / cache.WIKTEXTRACT_NAME
    if not input_path.exists():
        raise typer.BadParameter(f"Nothing parsed at {input_path}; parse it first")

    extractor = WiktionaryExtractor(
        language,
        frozenset(pos) if pos else None,
        minimum_year,
        maximum_year,
        open_locator(engine, processes, batch_size, gpu=gpu),
    )

    writer: Writer[Lemma] = open_writer(output_path)

    with writer:
        for lemma in extractor.extract(input_path):
            writer.write(lemma)

    typer.echo(f"Collected {output_path}")
