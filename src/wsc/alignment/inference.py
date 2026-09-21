"""Generate decisions through offline vLLM inference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from logging import getLogger
from typing import Protocol, cast

from ..models.alignment import (
    LanguageModel,
    ModelOutcome,
    ModelRequest,
    ModelSettings,
)

_LOGGER = getLogger(__name__)


class _ChatRequest(Protocol):
    """Describe the vLLM chat request fields used during inference."""

    messages: Sequence[dict[str, str]]
    chat_template_kwargs: dict[str, object] | None


class _Completion(Protocol):
    """Describe one generated completion returned by vLLM."""

    finish_reason: str | None
    text: str
    token_ids: Sequence[int]


class _RequestOutput(Protocol):
    """Describe the completion list returned for one request."""

    outputs: Sequence[_Completion]


class _Tokenizer(Protocol):
    """Describe the tokenizer operation used by Harmony parsing."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """
        Decode generated token identifiers.

        Args:
            token_ids: Generated token identifiers in sequence order.
            skip_special_tokens: Whether decoding omits special tokens.

        Returns:
            The decoded text.
        """
        ...


class _ReasoningParser(Protocol):
    """Describe the reasoning parser operations used by the adapter."""

    def extract_content_ids(
        self,
        input_ids: list[int],
    ) -> list[int]:
        """
        Extract Harmony final-channel token identifiers.

        Args:
            input_ids: Generated token identifiers including channel markers.

        Returns:
            Token identifiers belonging to the final response channel.
        """
        ...

    def extract_reasoning(
        self,
        model_output: str,
        request: _ChatRequest,
    ) -> tuple[str | None, str | None]:
        """
        Separate reasoning from final content.

        Args:
            model_output: Generated completion containing reasoning and final content.
            request: Chat context used by the parser.

        Returns:
            Reasoning and final content, each None when absent.
        """
        ...


class _ReasoningParserFactory(Protocol):
    """Construct one reasoning parser for each completion."""

    def __call__(
        self,
        *,
        tokenizer: _Tokenizer,
        chat_template_kwargs: dict[str, object] | None,
    ) -> _ReasoningParser:
        """
        Build a parser with request-specific context.

        Args:
            tokenizer: Tokenizer associated with the generation engine.
            chat_template_kwargs: Request-specific chat template arguments, or None.

        Returns:
            A reasoning parser configured for the request.
        """
        ...


class _ParserManager(Protocol):
    """Resolve the parser registered in the vLLM configuration."""

    @staticmethod
    def get_reasoning_parser(
        name: str,
    ) -> _ReasoningParserFactory:
        """
        Return the registered reasoning parser factory.

        Args:
            name: Parser name registered in the engine configuration.

        Returns:
            The factory for the named reasoning parser.
        """
        ...


class _StructuredOutputsConfiguration(Protocol):
    """Describe the configured reasoning parser name."""

    reasoning_parser: str | None


class _VllmConfiguration(Protocol):
    """Describe the vLLM configuration fields used by the adapter."""

    structured_outputs_config: _StructuredOutputsConfiguration


class _Engine(Protocol):
    """Describe the vLLM engine configuration boundary."""

    vllm_config: _VllmConfiguration


class _LanguageModel(Protocol):
    """Describe the offline vLLM operations used by the adapter."""

    llm_engine: _Engine

    def get_tokenizer(
        self,
    ) -> _Tokenizer:
        """
        Return the model tokenizer.

        Returns:
            The tokenizer associated with the loaded model.
        """
        ...

    def chat(
        self,
        *,
        messages: Sequence[Sequence[dict[str, str]]],
        sampling_params: Sequence[object],
        chat_template_kwargs: dict[str, object] | None,
        use_tqdm: bool,
    ) -> Sequence[_RequestOutput]:
        """
        Generate a batch of chat completions.

        Args:
            messages: Conversations submitted in batch order.
            sampling_params: Generation parameters for each conversation.
            chat_template_kwargs: Template arguments shared by the batch, or None.
            use_tqdm: Whether the engine displays a progress bar.

        Returns:
            Generated request outputs in submission order.
        """
        ...


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

        Raises:
            RuntimeError: If the offline vLLM backend is unavailable.
        """
        _LOGGER.info("Initializing model %s", settings.model)

        try:
            vllm = import_module("vllm")
            reasoning = import_module("vllm.reasoning")
        except ImportError as error:
            raise RuntimeError(
                "Offline alignment requires vLLM; install the platform backend first"
            ) from error

        model_factory = cast(Callable[..., _LanguageModel], vllm.LLM)
        parser_manager = cast(
            type[_ParserManager],
            reasoning.ReasoningParserManager,
        )

        engine_options = dict(settings.engine_options)

        if settings.reasoning_parser is not None:
            engine_options["reasoning_parser"] = settings.reasoning_parser

        self._llm: _LanguageModel = model_factory(
            model=settings.model,
            **engine_options,
        )

        self._tokenizer: _Tokenizer = self._llm.get_tokenizer()

        self._chat_template_kwargs: dict[str, object] = dict(
            settings.chat_template_options,
        )

        if settings.reasoning_effort is not None:
            self._chat_template_kwargs["reasoning_effort"] = settings.reasoning_effort

        configuration = self._llm.llm_engine.vllm_config
        parser_name = configuration.structured_outputs_config.reasoning_parser

        self._reasoning_parser_class: _ReasoningParserFactory | None = (
            parser_manager.get_reasoning_parser(parser_name) if parser_name else None
        )

        self._settings: ModelSettings = settings

        _LOGGER.info("Initialized model %s", settings.model)

    def _parse_reasoning(
        self,
        completion: _Completion,
        request: _ChatRequest,
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

        reasoning_parser = self._reasoning_parser_class(
            tokenizer=self._tokenizer,
            chat_template_kwargs=request.chat_template_kwargs,
        )

        harmony = import_module("vllm.reasoning.gptoss_reasoning_parser")
        harmony_parser = cast(
            type[object],
            harmony.GptOssReasoningParser,
        )

        if issubclass(
            cast(type[object], self._reasoning_parser_class),
            harmony_parser,
        ):
            return self._tokenizer.decode(
                reasoning_parser.extract_content_ids(list(completion.token_ids)),
                skip_special_tokens=True,
            )

        _, text = reasoning_parser.extract_reasoning(completion.text, request)

        return text

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
        [outcome] = self.generate_batch((request,))

        if outcome.text is None:
            raise ValueError(outcome.error)

        return outcome.text

    def _build_chat_request(
        self,
        request: ModelRequest,
    ) -> _ChatRequest:
        """
        Build one chat request from an alignment prompt.

        Args:
            request: Alignment prompt and response schema.

        Returns:
            Chat request carrying the configured template arguments.
        """
        protocol = import_module("vllm.entrypoints.openai.chat_completion.protocol")

        request_factory = cast(
            Callable[..., _ChatRequest],
            protocol.ChatCompletionRequest,
        )

        return request_factory(
            model=self._settings.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            chat_template_kwargs=self._chat_template_kwargs or None,
        )

    def _collect(
        self,
        output: _RequestOutput,
        request: _ChatRequest,
    ) -> ModelOutcome:
        """
        Read one generation without failing the remaining batch.

        Args:
            output: Engine output for one request.
            request: Chat request supplying parser context.

        Returns:
            Final JSON text or its failure description.
        """
        completion = output.outputs[0]

        if completion.finish_reason != "stop":
            return ModelOutcome(
                error="Incomplete model response\n"
                + f"Finish reason: {completion.finish_reason}\n\n"
                + f"Response\n{completion.text}",
            )

        text = self._parse_reasoning(completion, request)

        if text is None or not text.strip():
            return ModelOutcome(
                error=f"Empty final model response\n\nResponse\n{completion.text}",
            )

        return ModelOutcome(text.strip())

    def generate_batch(
        self,
        requests: Sequence[ModelRequest],
    ) -> tuple[ModelOutcome, ...]:
        """
        Generate structured completions in one engine call.

        Args:
            requests: Alignment prompts and response schemas.

        Returns:
            One outcome per request, in submission order.
        """
        if not requests:
            return ()

        sampling = import_module("vllm.sampling_params")

        sampling_factory = cast(
            Callable[..., object],
            sampling.SamplingParams,
        )

        structured_factory = cast(
            Callable[..., object],
            sampling.StructuredOutputsParams,
        )

        chat_requests = [self._build_chat_request(request) for request in requests]

        outputs = self._llm.chat(
            messages=[list(request.messages) for request in chat_requests],
            sampling_params=[
                sampling_factory(
                    temperature=self._settings.temperature,
                    max_tokens=self._settings.maximum_tokens,
                    structured_outputs=structured_factory(json=request.schema),
                    skip_special_tokens=False,
                )
                for request in requests
            ],
            chat_template_kwargs=self._chat_template_kwargs or None,
            use_tqdm=False,
        )

        return tuple(
            self._collect(output, request)
            for output, request in zip(outputs, chat_requests, strict=True)
        )


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
