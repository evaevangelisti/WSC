"""Generate decisions through offline vLLM inference."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..models.alignment import (
    LanguageModel,
    ModelOutcome,
    ModelRequest,
    ModelSettings,
)
from ..progress import nested_position, refresh_interval

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.reasoning import ReasoningParser
    from vllm.tokenizers import TokenizerLike

_LOGGER = getLogger(__name__)


class OfflineModel:
    """Use the offline vLLM engine for local models."""

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        """
        Configure the model engine and its output parser.

        Args:
            settings: Model and generation configuration.
        """
        _LOGGER.info("Initializing model %s", settings.model)

        try:
            from vllm import LLM
            from vllm.reasoning import ReasoningParserManager
        except ImportError as error:
            raise RuntimeError(
                "Offline alignment requires vLLM; install the platform backend first"
            ) from error

        engine_options = dict(settings.engine_options)

        if settings.reasoning_parser is not None:
            engine_options["reasoning_parser"] = settings.reasoning_parser

        self._llm: LLM = LLM(model=settings.model, **engine_options)
        self._tokenizer: TokenizerLike = self._llm.get_tokenizer()

        self._chat_template_kwargs: dict[str, object] = dict(
            settings.chat_template_options,
        )

        if settings.reasoning_effort is not None:
            self._chat_template_kwargs["reasoning_effort"] = settings.reasoning_effort

        configuration = self._llm.llm_engine.vllm_config
        parser_name = configuration.structured_outputs_config.reasoning_parser

        self._reasoning_parser_class: type[ReasoningParser] | None = (
            ReasoningParserManager.get_reasoning_parser(parser_name)
            if parser_name
            else None
        )

        self._settings: ModelSettings = settings

        _LOGGER.info("Initialized model %s", settings.model)

    def _parse_reasoning(
        self,
        completion: CompletionOutput,
        request: ChatCompletionRequest,
    ) -> str | None:
        """
        Extract final content using the configured reasoning parser.

        Args:
            completion: Generated text and token identifiers.
            request: Chat request supplying the parser context.

        Returns:
            Final content, or None when the parser finds no final answer.
        """
        if self._reasoning_parser_class is None:
            return completion.text

        from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser

        reasoning_parser = self._reasoning_parser_class(
            tokenizer=self._tokenizer,
            chat_template_kwargs=request.chat_template_kwargs,
        )

        if issubclass(self._reasoning_parser_class, GptOssReasoningParser):
            return self._tokenizer.decode(
                reasoning_parser.extract_content_ids(list(completion.token_ids)),
                skip_special_tokens=True,
            )

        _, text = reasoning_parser.extract_reasoning(completion.text, request)

        return text

    def _build_chat_request(
        self,
        request: ModelRequest,
    ) -> ChatCompletionRequest:
        """
        Build the chat request carrying one alignment prompt.

        Args:
            request: Alignment prompt and response schema.

        Returns:
            A chat request with the configured template arguments.
        """
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        return ChatCompletionRequest(
            model=self._settings.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            chat_template_kwargs=self._chat_template_kwargs or None,
        )

    def _collect(
        self,
        output: RequestOutput,
        chat_request: ChatCompletionRequest,
    ) -> ModelOutcome:
        """
        Extract the final content of one completed generation.

        Args:
            output: Engine output for a single request.
            chat_request: Chat request supplying the parser context.

        Returns:
            The final JSON text, or the reason generation failed.
        """
        completion = output.outputs[0]

        if completion.finish_reason != "stop":
            return ModelOutcome(
                error="Incomplete model response\n"
                + f"Finish reason: {completion.finish_reason}\n\n"
                + f"Response\n{completion.text}",
            )

        text = self._parse_reasoning(completion, chat_request)

        if text is None or not text.strip():
            return ModelOutcome(
                error=f"Empty final model response\n\nResponse\n{completion.text}",
            )

        return ModelOutcome(text.strip())

    def generate_many(
        self,
        requests: Sequence[ModelRequest],
    ) -> tuple[ModelOutcome, ...]:
        """
        Generate a complete batch of structured completions in one engine call.

        The engine schedules every prompt together, so continuous batching keeps
        the accelerators saturated instead of serving one request at a time.

        Args:
            requests: Alignment prompts and response schemas.

        Returns:
            One outcome per request, in submission order.
        """
        from vllm.sampling_params import SamplingParams, StructuredOutputsParams

        if not requests:
            return ()

        chat_requests = [self._build_chat_request(request) for request in requests]

        outputs = self._llm.chat(
            messages=[list(chat.messages) for chat in chat_requests],
            sampling_params=[
                SamplingParams(
                    temperature=self._settings.temperature,
                    max_tokens=self._settings.maximum_tokens,
                    structured_outputs=StructuredOutputsParams(json=request.schema),
                    skip_special_tokens=False,
                )
                for request in requests
            ],
            chat_template_kwargs=self._chat_template_kwargs or None,
            use_tqdm=partial(
                tqdm,
                position=nested_position(),
                leave=False,
                mininterval=refresh_interval(),
            ),
        )

        return tuple(
            self._collect(output, chat_request)
            for output, chat_request in zip(outputs, chat_requests, strict=True)
        )

    def generate(
        self,
        request: ModelRequest,
    ) -> str:
        """
        Generate and parse a structured completion without an API server.

        Args:
            request: Alignment prompt and response schema.

        Returns:
            Final JSON text without reasoning content.

        Raises:
            ValueError: If generation is incomplete or has no final text.
        """
        [outcome] = self.generate_many((request,))

        if outcome.text is None:
            raise ValueError(outcome.error)

        return outcome.text


def open_model(
    settings: ModelSettings,
) -> LanguageModel:
    """
    Construct an offline engine for the configured language model.

    Args:
        settings: Model identifier and generation options.

    Returns:
        An offline vLLM model.
    """
    return OfflineModel(settings)
