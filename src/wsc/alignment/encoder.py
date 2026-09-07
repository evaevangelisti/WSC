"""
Cross-encoder inference for lexical alignment hypotheses.
"""

from collections.abc import Sequence

from ..constants import (
    ALIGNMENT_BATCH_SIZE,
    ALIGNMENT_MAXIMUM_LENGTH,
    ALIGNMENT_MODEL,
    DEFAULT_INSTRUCTIONS,
)
from ..models.alignment import AlignmentInstructions, Comparison, Reranker


class CrossEncoderScorer:
    """Batch semantic hypotheses through a locally executable reranker."""

    def __init__(
        self,
        model: str = ALIGNMENT_MODEL,
        *,
        revision: str = "main",
        device: str | None = None,
        batch_size: int = ALIGNMENT_BATCH_SIZE,
        maximum_length: int = ALIGNMENT_MAXIMUM_LENGTH,
        instructions: AlignmentInstructions = DEFAULT_INSTRUCTIONS,
    ) -> None:
        """
        Load model weights and replace retrieval instructions with relation judgement.

        Args:
            model: Hugging Face identifier or local model directory.
            revision: Model revision, preferably an immutable commit.
            device: Torch device or automatic selection.
            batch_size: Number of candidate pairs per inference batch.
            maximum_length: Maximum token count per pair.
            instructions: Selected general and relation-specific instructions.
        """
        from sentence_transformers import CrossEncoder
        from torch.nn import Identity

        self._model: Reranker = CrossEncoder(
            model,
            revision=revision,
            device=device,
            max_length=maximum_length,
            prompts={"alignment": instructions.instruction},
            default_prompt_name="alignment",
            activation_fn=Identity(),
            model_kwargs={"torch_dtype": "auto"},
        )

        self._batch_size: int = batch_size

    def score(
        self,
        pairs: Sequence[Comparison],
    ) -> list[float]:
        """
        Return raw relation scores without interpreting them as probabilities.

        Args:
            pairs: Hypotheses and definitions in candidate order.

        Returns:
            One uncalibrated score per pair.
        """
        if not pairs:
            return []

        scores = self._model.predict(
            [(pair.query, pair.document) for pair in pairs],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return [float(score) for score in scores]
