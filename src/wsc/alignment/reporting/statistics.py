"""Count resolved senses and associations for each alignment task."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ...models.alignment import AlignmentResult, AlignmentTask


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
