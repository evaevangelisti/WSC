"""
Generate decisions through offline vLLM inference.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from importlib import import_module
from logging import getLogger
from typing import cast

from ...models.alignment import (
    LanguageModel,
    ModelOutcome,
    ModelRequest,
    ModelSettings,
)
from .protocols import (
    ChatRequest,
    Completion,
    ParserManager,
    ReasoningParserFactory,
    RequestOutput,
    Tokenizer,
    VllmLanguageModel,
    XGrammarModule,
)

_LOGGER = getLogger(__package__)


class OfflineModel:
    """
    Use the offline vLLM engine for local models.
    """

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

        model_factory = cast(Callable[..., VllmLanguageModel], vllm.LLM)
        parser_manager = cast(
            type[ParserManager],
            reasoning.ReasoningParserManager,
        )

        engine_options = dict(settings.engine_options)

        if settings.reasoning_parser is not None:
            engine_options["reasoning_parser"] = settings.reasoning_parser

        self._llm: VllmLanguageModel = model_factory(
            model=settings.model,
            **engine_options,
        )

        self._tokenizer: Tokenizer = self._llm.get_tokenizer()

        self._chat_template_kwargs: dict[str, object] = dict(
            settings.chat_template_options,
        )

        if settings.reasoning_effort is not None:
            self._chat_template_kwargs["reasoning_effort"] = settings.reasoning_effort

        configuration = self._llm.llm_engine.vllm_config
        parser_name = configuration.structured_outputs_config.reasoning_parser

        self._reasoning_parser_class: ReasoningParserFactory | None = (
            parser_manager.get_reasoning_parser(parser_name) if parser_name else None
        )

        self._settings: ModelSettings = settings

        _LOGGER.info("Initialized model %s", settings.model)

    def _parse_reasoning(
        self,
        completion: Completion,
        request: ChatRequest,
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

    def _build_chat_request(
        self,
        request: ModelRequest,
    ) -> ChatRequest:
        """
        Build one chat request from an alignment prompt.

        Args:
            request: Alignment prompt and response schema.

        Returns:
            Chat request carrying the configured template arguments.
        """
        protocol = import_module("vllm.entrypoints.openai.chat_completion.protocol")

        request_factory = cast(
            Callable[..., ChatRequest],
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
        output: RequestOutput,
        request: ChatRequest,
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

    def _report_invalid_schemas(
        self,
        requests: Sequence[ModelRequest],
    ) -> None:
        """
        Print records whose schemas fail after the inference engine dies.

        Args:
            requests: Submitted prompts and schemas in engine order.
        """
        xgrammar = cast(XGrammarModule, cast(object, import_module("xgrammar")))

        failures = 0

        for index, request in enumerate(requests):
            try:
                _ = xgrammar.Grammar.from_json_schema(request.schema)
            except RuntimeError as error:
                failures += 1

                _LOGGER.error(
                    "XGrammar rejected request %s: %s\nPrompt:\n%s\nSchema:\n%s",
                    index,
                    error,
                    request.prompt,
                    json.dumps(request.schema, ensure_ascii=False, indent=2),
                )

        if not failures:
            _LOGGER.error(
                "XGrammar could not reproduce a schema error in %s failed requests",
                len(requests),
            )

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

        engine_exceptions = import_module("vllm.v1.engine.exceptions")
        engine_dead_error = cast(
            type[Exception],
            engine_exceptions.EngineDeadError,
        )

        chat_requests = [self._build_chat_request(request) for request in requests]

        try:
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
                use_tqdm=True,
            )
        except engine_dead_error:
            self._report_invalid_schemas(requests)

            raise

        return tuple(
            self._collect(output, request)
            for output, request in zip(outputs, chat_requests, strict=True)
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
        [outcome] = self.generate_batch((request,))

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
