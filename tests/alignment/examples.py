"""Build model responses and lexical queries for alignment tests."""

from collections.abc import Sequence

from wsc.models import POS
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentTask,
    Definition,
    ModelOutcome,
    ModelRequest,
)


class Model:
    """Return configured responses and retain the generated requests."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        """
        Store generated responses.

        Args:
            responses: Model responses in request order.
        """
        self.responses: list[str] = list(responses)
        self.requests: list[ModelRequest] = []

    def generate(
        self,
        request: ModelRequest,
    ) -> str:
        """
        Record a request and return its response.

        Args:
            request: Actual rendered model request.

        Returns:
            The next configured response.
        """
        self.requests.append(request)

        return self.responses.pop(0)

    def generate_many(
        self,
        requests: Sequence[ModelRequest],
    ) -> tuple[ModelOutcome, ...]:
        """
        Record a batch of requests and return their responses.

        Args:
            requests: Actual rendered model requests.

        Returns:
            The next configured response for each request, in order.
        """
        return tuple(ModelOutcome(self.generate(request)) for request in requests)


def build_query(
    task: AlignmentTask = AlignmentTask.TRANSLATIONS,
) -> AlignmentQuery:
    """
    Build a two-source query with overlapping contextual vocabulary.

    Args:
        task: Resource being aligned.

    Returns:
        Complete source and candidate definitions.
    """
    return AlignmentQuery(
        task,
        "word.noun",
        "word.noun",
        "word",
        POS.NOUN,
        (
            Definition("s1", ("parent", "first sense"), ("synonym",)),
            Definition("s2", ("second sense",)),
        ),
        (
            Definition("t1", ("first heading",), ("target synonym",)),
            Definition("t2", ("second heading",)),
        ),
    )


def build_decision(
    target: str | None,
    relation: str = "translation",
) -> list[dict[str, str]] | None:
    """
    Build a generated association or abstention.

    Args:
        target: Accepted candidate identifier.
        relation: Directed semantic relation.

    Returns:
        A supported association or null.
    """
    if target is None:
        return None

    return [
        {
            "target_id": target,
            "relation": relation,
            "reason": "The definitions express the same concept.",
        },
    ]
