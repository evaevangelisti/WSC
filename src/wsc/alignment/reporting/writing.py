"""
Publish aligned senses, task reports, and provenance together.
"""

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from ...constants import ALIGNMENT_MANIFEST, ALIGNMENT_REPORTS_DIR, ALIGNMENT_SENSES
from ...export.formats.jsonl import JSONLWriter
from ...files import Compression, compressed_name
from ...models import Lemma
from ...reporting import publish_files, stage_json
from .statistics import AlignmentStatistics


def write_alignment(
    entries: Iterable[Lemma],
    output_dir: Path,
    statistics: AlignmentStatistics,
    manifest: Mapping[str, object],
    compression: Compression | None = None,
) -> None:
    """
    Publish aligned senses, task reports, and provenance together.

    Args:
        entries: Aligned entries consumed once.
        output_dir: Destination for senses, reports, and provenance.
        statistics: Counts populated while entries are consumed.
        manifest: Source and configuration provenance.
        compression: Format used to compress the generated files.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".alignment-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)
        senses_name = compressed_name(ALIGNMENT_SENSES, compression)
        manifest_name = compressed_name(ALIGNMENT_MANIFEST, compression)
        staged_output = staging_dir / senses_name

        with JSONLWriter[Lemma](staged_output) as writer:
            for entry in entries:
                writer.write(entry)

        reports = statistics.reports()
        report_files = {
            task: (
                f"{ALIGNMENT_REPORTS_DIR}/"
                f"{compressed_name(f'{task}.json', compression)}"
            )
            for task in reports
        }

        completed_manifest = {
            **manifest,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": {
                "senses": senses_name,
                "reports": report_files,
                "manifest": manifest_name,
            },
            "totals": {task: report["senses"] for task, report in reports.items()},
        }

        staged_reports = staging_dir / ALIGNMENT_REPORTS_DIR
        staged_reports.mkdir()

        for task, report in reports.items():
            stage_json(staging_dir / report_files[task], report)

        stage_json(staging_dir / manifest_name, completed_manifest)

        publish_files(
            staging_dir,
            output_dir,
            (
                senses_name,
                *report_files.values(),
                manifest_name,
            ),
        )
