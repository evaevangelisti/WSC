"""
Write collected senses, statistics, and provenance together.
"""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ..archives import publish_archive
from ..constants import COLLECTION_FILES
from ..export.formats.jsonl import JSONLWriter
from ..files import Compression
from ..models import Lemma
from ..reporting import publish_files, stage_json
from .markdown import render_markdown
from .statistics import Statistics


def write_collection(
    entries: Iterable[Lemma],
    output_dir: Path,
    manifest: Mapping[str, object],
    compression: Compression | None = None,
) -> None:
    """
    Write the collection and its reports after extraction completes successfully.

    Files are staged together, then published after collection completes.

    Args:
        entries: Collected entries, consumed once.
        output_dir: Directory name used for output or its archive.
        manifest: Source and configuration provenance.
        compression: Format used to compress the output directory.
    """
    statistics = Statistics()
    files = COLLECTION_FILES

    started_at = datetime.now(UTC).isoformat()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".collection-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)

        with JSONLWriter[Lemma](staging_dir / files["senses"]) as writer:
            for entry in entries:
                writer.write(entry)
                statistics.add(entry)

        report = statistics.to_dict()

        provenance = {
            **manifest,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": files,
            "totals": report["totals"],
        }

        stage_json(staging_dir / files["statistics"], report)
        stage_json(staging_dir / files["manifest"], provenance)

        with (staging_dir / files["report"]).open("w", encoding="utf-8") as stream:
            _ = stream.write(render_markdown(statistics))

        if compression:
            publish_archive(staging_dir, output_dir, compression)

            return

        publish_files(staging_dir, output_dir, files.values())
