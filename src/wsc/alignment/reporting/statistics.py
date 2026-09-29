"""
Count resolved senses and associations for each alignment task.
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from ...models import Lemma
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

    def add_lemma(
        self,
        lemma: Lemma,
    ) -> None:
        """
        Count final synset associations after all resource passes.

        Args:
            lemma: Aligned entry after its last resource pass.
        """
        self.evaluated_senses += len(lemma.senses)

        self.aligned_senses += sum(bool(sense.synsets) for sense in lemma.senses)

        self.associations += sum(len(sense.synsets) for sense in lemma.senses)

        self.relations.update(
            association.relation
            for sense in lemma.senses
            for association in sense.synsets
        )


class AlignmentStatistics:
    """
    Accumulate one report for each requested alignment task.

    Reports are returned in the order in which tasks were requested.
    """

    def __init__(
        self,
        tasks: Iterable[AlignmentTask],
        *,
        count_final_synsets: bool = False,
    ) -> None:
        """
        Initialize empty task reports.

        Args:
            tasks: Requested alignment resources.
            count_final_synsets: Count final entries across multiple passes.
        """
        self._tasks: tuple[AlignmentTask, ...] = tuple(tasks)

        self._count_final_synsets: bool = count_final_synsets

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
        if result.query.task == AlignmentTask.SYNSETS and self._count_final_synsets:
            return

        self._statistics[result.query.task].add(result)

    def observe(
        self,
        entries: Iterable[Lemma],
    ) -> Iterator[Lemma]:
        """
        Count final synset senses while preserving the output stream.

        Args:
            entries: Aligned entries from every resource pass.

        Yields:
            Original entries in their output order.
        """
        for lemma in entries:
            if self._count_final_synsets:
                self._statistics[AlignmentTask.SYNSETS].add_lemma(lemma)

            yield lemma

    def reports(
        self,
    ) -> dict[AlignmentTask, dict[str, object]]:
        """
        Return reports in requested task order.

        Returns:
            JSON-compatible report data keyed by alignment task.
        """
        return {task: self._statistics[task].to_dict(task) for task in self._tasks}
