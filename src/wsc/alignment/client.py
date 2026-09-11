"""Generate decisions through offline vLLM batch inference."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM

from ..models.alignment import LanguageModel, ModelRequest, ModelSettings


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
        except ImportError as error:
            raise RuntimeError(
                "Offline alignment requires vLLM; install the platform backend first"
            ) from error

        self._llm: LLM = LLM(
            model=settings.model,
            **dict(settings.engine_options),
        )

        self._settings: ModelSettings = settings

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
        from vllm.sampling_params import GuidedDecodingParams

        sampling_params = SamplingParams(
            temperature=self._settings.temperature,
            max_tokens=self._settings.maximum_tokens,
            guided_decoding=GuidedDecodingParams(json=request.schema),
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

        if completion.finish_reason != "stop" or not completion.text:
            raise ValueError(f"Incomplete model response: {completion.finish_reason}")

        return completion.text


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
