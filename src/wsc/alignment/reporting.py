"""Build alignment reports and publish them with the aligned collection."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory

from ..constants import ALIGNMENT_MANIFEST, ALIGNMENT_REPORTS_DIR, ALIGNMENT_SENSES
from ..export.formats.jsonl import JSONLWriter
from ..models import Lemma
from ..models.alignment import (
    AlignmentPrompts,
    AlignmentResult,
    AlignmentTask,
    GlossMode,
    ModelSettings,
)
from ..reporting import describe_source, publish_files, stage_json


@dataclass(slots=True)
class _TaskStatistics:
    """
    Accumulate decision and relation counts for one task.

    Counts include evaluated senses, accepted associations, and their relations.
    """

    evaluated_senses: int = 0

    aligned_senses: int = 0

    associations: int = 0
    relations: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        result: AlignmentResult,
    ) -> None:
        """
        Count one complete or partially reused result.

        Args:
            result: Resolved source decisions.
        """
        self.evaluated_senses += len(result.decisions)

        self.aligned_senses += sum(
            bool(decision.links) for decision in result.decisions
        )

        self.associations += len(result.links)
        self.relations.update(link.relation for link in result.links)

    def to_dict(
        self,
        task: AlignmentTask,
    ) -> dict[str, object]:
        """
        Export JSON-compatible task statistics.

        Args:
            task: Alignment resource described by the counts.

        Returns:
            Sense totals and the relation distribution.
        """
        return {
            "task": task,
            "senses": {
                "evaluated": self.evaluated_senses,
                "aligned": self.aligned_senses,
                "unaligned": self.evaluated_senses - self.aligned_senses,
            },
            "associations": self.associations,
            "relations": dict(sorted(self.relations.items())),
        }


class AlignmentStatistics:
    """
    Accumulate one report for each requested alignment task.

    Reports are returned in the order in which tasks were requested.
    """

    def __init__(
        self,
        tasks: Iterable[AlignmentTask],
    ) -> None:
        """
        Initialize empty task reports.

        Args:
            tasks: Requested alignment resources.
        """
        self._tasks: tuple[AlignmentTask, ...] = tuple(tasks)
        self._statistics: dict[AlignmentTask, _TaskStatistics] = {
            task: _TaskStatistics() for task in self._tasks
        }

    def add(
        self,
        result: AlignmentResult,
    ) -> None:
        """
        Add resolved decisions to their task report.

        Args:
            result: Complete or partially reused result.
        """
        self._statistics[result.query.task].add(result)

    def reports(
        self,
    ) -> dict[AlignmentTask, dict[str, object]]:
        """
        Return reports in requested task order.

        Returns:
            JSON-compatible report data keyed by alignment task.
        """
        return {task: self._statistics[task].to_dict(task) for task in self._tasks}


def build_manifest(
    input_path: Path,
    settings: ModelSettings,
    mode: GlossMode,
    prompts: AlignmentPrompts,
    tasks: tuple[AlignmentTask, ...],
    wordnet_edition: str | None,
) -> dict[str, object]:
    """
    Describe the source and configuration of an alignment run.

    Args:
        input_path: Collected senses supplied to alignment.
        settings: Language model generation settings.
        mode: Wiktionary definition representation.
        prompts: Selected prompt collection.
        tasks: Requested alignment resources.
        wordnet_edition: Candidate edition, or None when unused.

    Returns:
        Provenance completed after alignment succeeds.
    """
    return {
        "source": describe_source(input_path),
        "tasks": list(tasks),
        "settings": {
            **asdict(settings),
            "gloss_mode": mode,
            "prompts": prompts.name,
            "wordnet_edition": wordnet_edition,
        },
        "runtime": {
            "python": python_version(),
            "packages": {"wsc": version("wsc")},
        },
        "started_at": datetime.now(UTC).isoformat(),
    }


def write_alignment(
    entries: Iterable[Lemma],
    output_dir: Path,
    statistics: AlignmentStatistics,
    manifest: Mapping[str, object],
) -> None:
    """
    Publish aligned senses, task reports, and provenance together.

    Args:
        entries: Aligned entries consumed once.
        output_dir: Destination for senses, reports, and provenance.
        statistics: Counts populated while entries are consumed.
        manifest: Source and configuration provenance.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".alignment-", dir=output_dir.parent) as directory:
        staging_dir = Path(directory)
        staged_output = staging_dir / ALIGNMENT_SENSES

        with JSONLWriter[Lemma](staged_output) as writer:
            for entry in entries:
                writer.write(entry)

        reports = statistics.reports()
        report_files = {
            task: f"{ALIGNMENT_REPORTS_DIR}/{task}.json" for task in reports
        }
        completed_manifest = {
            **manifest,
            "completed_at": datetime.now(UTC).isoformat(),
            "files": {
                "senses": ALIGNMENT_SENSES,
                "reports": report_files,
                "manifest": ALIGNMENT_MANIFEST,
            },
            "totals": {task: report["senses"] for task, report in reports.items()},
        }

        staged_reports = staging_dir / ALIGNMENT_REPORTS_DIR
        staged_reports.mkdir()

        for task, report in reports.items():
            stage_json(staged_reports / f"{task}.json", report)

        stage_json(staging_dir / ALIGNMENT_MANIFEST, completed_manifest)

        publish_files(
            staging_dir,
            output_dir,
            (
                ALIGNMENT_SENSES,
                *(f"{ALIGNMENT_REPORTS_DIR}/{task}.json" for task in reports),
                ALIGNMENT_MANIFEST,
            ),
        )
