"""
Command line for collecting senses out of Wiktionary.
"""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from .constants import CHUNK_SIZE, TIMEOUT, USER_AGENT
from .export import open_writer
from .extract import WiktionaryExtractor
from .models import POS
from .upstream import cache, download, repositories, wiktextract

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
        date = repositories.wiktionary.latest_date(language, user_agent, TIMEOUT)
        typer.echo(f"Resolved latest to {date}")

    dump_path = cache.dump_dir(cache_dir, language, date) / cache.DUMP_NAME
    if dump_path.exists():
        typer.echo(f"Already fetched {dump_path}")
        return

    download(
        repositories.wiktionary.url(language, date),
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

    skipped_lines = wiktextract.parse(dump_path, output_path, language, processes)

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
    )

    with open_writer(output_path) as writer:
        for lemma in extractor.extract(input_path):
            writer.write(lemma)

    typer.echo(f"Collected {output_path}")


@app.command()
def wordnet(
    wordnet_version: Annotated[
        str,
        typer.Option(
            envvar="WSC_WORDNET_VERSION",
            help="Wordnet edition to use, as 2025, or latest.",
        ),
    ] = cache.LATEST,
    cache_dir: CacheDir = None,
) -> None:
    """
    Download the wordnet the senses are aligned with. Needs the network.
    """
    user_agent = USER_AGENT.format(version=version("wsc"))

    edition = wordnet_version
    if edition == cache.LATEST:
        edition = repositories.wordnet.latest_version(user_agent, TIMEOUT)
        typer.echo(f"Resolved latest to {edition}")

    wordnet_path = cache.wordnet_path(cache_dir, edition)
    if wordnet_path.exists():
        typer.echo(f"Already fetched {wordnet_path}")
        return

    download(
        repositories.wordnet.url(edition),
        wordnet_path,
        user_agent,
        TIMEOUT,
        CHUNK_SIZE,
    )

    typer.echo(f"Fetched {wordnet_path}")
