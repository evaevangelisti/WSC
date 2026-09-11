"""Generate decisions through offline vLLM batch inference."""

from collections.abc import Sequence
from typing import Protocol

from ..models.alignment import LanguageModel, ModelRequest, ModelSettings


class Completion(Protocol):
    """Provide the vLLM completion fields used by the adapter."""

    finish_reason: str | None
    text: str
    token_ids: Sequence[int]


class ChatModel:
    """Use the offline vLLM engine for local models."""

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        """
        Configure the model enigne.

        Args:
            settings: Model and generation configuration.
        """
        try:
            from vllm import LLM
            from vllm.reasoning import ReasoningParserManager
        except ImportError as error:
            raise RuntimeError(
                "Offline alignment requires vLLM; install the platform backend first"
            ) from error

        self._llm: LLM = LLM(
            model=settings.model,
            **dict(settings.engine_options),
        )

        self._tokenizer = self._llm.get_tokenizer()

        parser = ReasoningParserManager.get_reasoning_parser("openai_gptoss")
        self._reasoning_parser = parser(tokenizer=self._tokenizer)

        self._settings: ModelSettings = settings

    def _parse_reasoning(
        self,
        completion: Completion,
    ) -> str:
        """
        Extract and decode the final response from a reasoning completion.

        Args:
            completion: vLLM completion containing generated token IDs.

        Returns:
            The decoded final response without reasoning content.

        Raises:
            ValueError: If the completion has no final response.
        """
        token_ids = completion.token_ids
        content_ids = self._reasoning_parser.extract_content_ids(token_ids)

        text = self._tokenizer.decode(content_ids).strip()

        if not text:
            raise ValueError(f"Empty final model response: text={completion.text!r}")

        return text

    def generate(
        self,
        request: ModelRequest,
    ) -> str:
        """
        Request a structured chat completion.

        Args:
            request: Alignment prompt and response schema.

        Returns:
            Generated JSON text.

        Raises:
            ValueError: If generation is incomplete or has no text.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sampling_params = SamplingParams(
            temperature=self._settings.temperature,
            max_tokens=self._settings.maximum_tokens,
            structured_outputs=StructuredOutputsParams(json=request.schema),
        )

        chat_template_kwargs = (
            {"reasoning_effort": self._settings.reasoning_effort}
            if self._settings.reasoning_effort is not None
            else None
        )

        [output] = self._llm.chat(
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )

        completion = output.outputs[0]

        if completion.finish_reason != "stop":
            raise ValueError(
                "Incomplete model response: "
                + f"finish_reason={completion.finish_reason!r}, "
                + f"text={completion.text!r}"
            )

        return self._parse_reasoning(completion)


def open_model(
    settings: ModelSettings,
) -> LanguageModel:
    """
    Construct an offline engine for the configured language model.

    Args:
        settings: Model identifier and generation options.

    Returns:
        An offline vLLM inference client.
    """
    return ChatModel(settings)
