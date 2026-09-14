"""Record the sources and settings used to collect a dump."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from platform import python_version

from ..constants import LANGUAGE, SPACY_PIPELINE
from ..models import POS, Engine


@dataclass(frozen=True, slots=True)
class CollectionSettings:
    """
    Filters and offset extraction options for a collection.

    Attributes:
        parts_of_speech: Parts of speech retained in the output.
        minimum_year: Oldest retained quotation year, or None.
        maximum_year: Newest retained quotation year, or None.
        engine: Offset extraction engine.
        processes: Requested worker count; Stanza always uses one.
        batch_size: Sentences per engine batch.
        gpu: Whether GPU inference is requested.
    """

    parts_of_speech: tuple[POS, ...]
    minimum_year: int | None
    maximum_year: int | None
    engine: Engine
    processes: int
    batch_size: int
    gpu: bool


def _describe_source(
    path: Path,
) -> dict[str, object]:
    """
    Record the location, size, and modification time of a source file.

    Args:
        path: Source file used by the collection.

    Returns:
        Absolute path, byte count, and modification timestamp in UTC.

    Raises:
        OSError: If the source file metadata cannot be read.
    """
    status = path.stat()

    return {
        "path": str(path.resolve()),
        "bytes": status.st_size,
        "modified_at": datetime.fromtimestamp(status.st_mtime, UTC).isoformat(),
    }


def build_manifest(
    input_path: Path,
    off_page_path: Path | None,
    dump_date: str,
    settings: CollectionSettings,
) -> dict[str, object]:
    """
    Describe the parsed sources and requested collection configuration.

    Args:
        input_path: Parsed Wiktextract entries.
        off_page_path: Supplemental translations, or None when unavailable.
        dump_date: Resolved dump date.
        settings: Filters and engine options used by the extractor.

    Returns:
        Provenance stored beside the completed collection.
    """
    return {
        "dump_date": dump_date,
        "language": LANGUAGE,
        "sources": {
            "wiktextract": _describe_source(input_path),
            "off_page_translations": _describe_source(off_page_path)
            if off_page_path
            else None,
        },
        "settings": asdict(settings),
        "runtime": {
            "python": python_version(),
            "packages": {
                name: version(name) for name in ("wsc", "kwic", "wiktextract")
            },
            "spacy_pipeline": SPACY_PIPELINE
            if settings.engine == Engine.SPACY
            else None,
            "effective_processes": 1
            if settings.engine == Engine.STANZA
            else settings.processes,
        },
    }
