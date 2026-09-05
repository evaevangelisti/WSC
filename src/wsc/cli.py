"""
Command line for collecting senses out of Wiktionary.
"""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from .constants import (
    BATCH_SIZE,
    CHUNK_SIZE,
    KAIKKI_URL,
    LANGUAGE_SECTION,
    PROCESSES,
    TIMEOUT,
    USER_AGENT,
)
from .export import Writer, open_writer
from .extract import (
    DumpExtractor,
    WiktionaryExtractor,
    WordNetExtractor,
    build_off_page_translations,
    narrow,
    open_locator,
    read_entries,
    read_off_page_translations,
    write_off_page_translations,
)
from .models import POS, Engine, Lemma, Synset
from .upstream import cache, download, repositories, wiktextract

# Named apart from the signatures, so that the commands asking for the same
# option share one.

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
    dump_date: DumpDate = cache.LATEST,
    cache_dir: CacheDir = None,
) -> None:
    """
    Download a Wiktionary dump. Needs the network.
    """
    user_agent = USER_AGENT.format(version=version("wsc"))

    date = dump_date
    if date == cache.LATEST:
        date = repositories.wiktionary.latest_date(user_agent, TIMEOUT)
        typer.echo(f"Resolved latest to {date}")

    dump_path = cache.dump_dir(cache_dir, date) / cache.DUMP_NAME
    if dump_path.exists():
        typer.echo(f"Already fetched {dump_path}")
        return

    download(
        repositories.wiktionary.url(date),
        dump_path,
        user_agent,
        TIMEOUT,
        CHUNK_SIZE,
    )

    typer.echo(f"Fetched {dump_path}")


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
        typer.echo(f"Already parsed {output_path}")
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
            typer.echo(f"Set aside {skipped_lines} lines holding no entry")

    off_page_translations = build_off_page_translations(
        DumpExtractor(LANGUAGE_SECTION).extract(dump_path),
        read_entries(output_path, "Answering the pointers"),
    )

    write_off_page_translations(off_page_translations_path, off_page_translations)

    translated = len(off_page_translations)
    typer.echo(f"Parsed {output_path}, {translated} entries translated elsewhere")


@app.command()
def collect(
    output_path: Annotated[
        Path,
        typer.Argument(
            metavar="output",
            help="Where the senses go; the suffix picks the format.",
        ),
    ],
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

    writer: Writer[Lemma] = open_writer(output_path)

    with writer:
        for lemma in extractor.extract(input_path):
            writer.write(lemma)

    typer.echo(f"Collected {output_path}")


@app.command()
def wordnet(
    edition: Annotated[
        str,
        typer.Option(
            help="Wordnet edition to use, as 2025, or latest.",
        ),
    ] = cache.LATEST,
    cache_dir: CacheDir = None,
) -> None:
    """
    Download the wordnet the senses are aligned with, and read its synsets.

    Needs the network, unless the edition asked for is already here.
    """
    user_agent = USER_AGENT.format(version=version("wsc"))

    if edition == cache.LATEST:
        edition = repositories.wordnet.latest_version(user_agent, TIMEOUT)
        typer.echo(f"Resolved latest to {edition}")

    wordnet_dir = cache.wordnet_dir(cache_dir, edition)

    wordnet_path = wordnet_dir / cache.WORDNET_NAME
    if wordnet_path.exists():
        typer.echo(f"Already fetched {wordnet_path}")
    else:
        download(
            repositories.wordnet.url(edition),
            wordnet_path,
            user_agent,
            TIMEOUT,
            CHUNK_SIZE,
        )

        typer.echo(f"Fetched {wordnet_path}")

    output_path = wordnet_dir / cache.SYNSETS_NAME
    if output_path.exists():
        typer.echo(f"Already read {output_path}")
        return

    # A cache answering to a filter is one the next run cannot trust.
    extractor = WordNetExtractor(None)

    writer: Writer[Synset] = open_writer(output_path)

    with writer:
        for synset in extractor.extract(wordnet_path):
            writer.write(synset)

    typer.echo(f"Read {output_path}")
