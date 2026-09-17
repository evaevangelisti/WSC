"""Write collected senses, statistics, and provenance together."""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ..constants import COLLECTION_FILES
from ..export.formats.jsonl import JSONLWriter
from ..models import Lemma
from ..reporting import publish_files, stage_json
from .markdown import render_markdown
from .statistics import Statistics


def write_collection(
    entries: Iterable[Lemma],
    output_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    """
    Write the collection and its reports after extraction completes successfully.

    Files are staged together, then replaced individually with the manifest last.

    Args:
        entries: Collected entries, consumed once.
        output_dir: Destination directory for the four collection files.
        manifest: Source and configuration provenance.
    """
    statistics = Statistics()

    started_at = datetime.now(UTC).isoformat()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".collection-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)

        with JSONLWriter[Lemma](staging_dir / COLLECTION_FILES["senses"]) as writer:
            for entry in entries:
                writer.write(entry)
                statistics.add(entry)

        report = statistics.to_dict()

        provenance = {
            **manifest,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": COLLECTION_FILES,
            "totals": report["totals"],
        }

        stage_json(staging_dir / COLLECTION_FILES["statistics"], report)
        stage_json(staging_dir / COLLECTION_FILES["manifest"], provenance)

        _ = (staging_dir / COLLECTION_FILES["report"]).write_text(
            render_markdown(statistics),
            encoding="utf-8",
        )

        publish_files(staging_dir, output_dir, COLLECTION_FILES.values())
