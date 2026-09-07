"""
Public cross-encoder adapter behavior without downloading model weights.
"""

from collections.abc import Sequence
from typing import Literal

import pytest
import sentence_transformers
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import CrossEncoderScorer
from wsc.constants import INSTRUCTIONS_PATH
from wsc.models import Reranker
from wsc.models.alignment import AlignmentInstructions, Comparison
from wsc.reading import read_instructions


@given(
    st.lists(st.integers(min_value=-100, max_value=100), max_size=12),
    st.sampled_from(read_instructions(INSTRUCTIONS_PATH)),
)
def test_adapter_preserves_pair_order_scores_and_inference_settings(
    monkeypatch: pytest.MonkeyPatch,
    values: list[int],
    instructions: AlignmentInstructions,
) -> None:
    """Batches preserve scores and skip prediction for empty inputs."""
    pairs = [Comparison(f"query {index}", "definition") for index in range(len(values))]
    configuration: dict[str, object] = {}
    received: list[tuple[str, str]] = []

    class Model:
        """External prediction boundary with one scalar output per pair."""

        def predict(
            self,
            inputs: Sequence[tuple[str, str]],
            *,
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: Literal[True],
        ) -> list[int]:
            """
            Record prediction arguments and return numeric evidence.

            Args:
                inputs: Ordered text pairs.
                batch_size: Requested inference batch size.
                show_progress_bar: Whether progress output is enabled.
                convert_to_numpy: Requested array output mode.

            Returns:
                Generated scalar scores in candidate order.
            """
            assert inputs
            assert batch_size == 3
            assert not show_progress_bar
            assert convert_to_numpy
            received.extend(inputs)

            return values

    def load_model(name: str, **options: object) -> Reranker:
        """
        Replace weight loading while preserving constructor arguments.

        Args:
            name: Requested model identifier.
            options: Model construction settings.

        Returns:
            A text-pair prediction implementation.
        """
        assert name == "local-model"
        configuration.update(options)

        return Model()

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", load_model)
    scorer = CrossEncoderScorer(
        "local-model",
        revision="fixed-revision",
        device="cpu",
        batch_size=3,
        maximum_length=128,
        instructions=instructions,
    )

    assert scorer.score(pairs) == [float(value) for value in values]
    assert received == [(pair.query, pair.document) for pair in pairs]
    assert configuration["revision"] == "fixed-revision"
    assert configuration["device"] == "cpu"
    assert configuration["max_length"] == 128
    assert configuration["prompts"] == {"alignment": instructions.instruction}
    assert configuration["default_prompt_name"] == "alignment"
