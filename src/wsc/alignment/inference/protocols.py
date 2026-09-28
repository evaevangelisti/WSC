"""
Type the vLLM and XGrammar operations used by offline inference.
"""

from collections.abc import Sequence
from typing import Protocol


class ChatRequest(Protocol):
    """
    Describe the vLLM chat request fields used during inference.
    """

    messages: Sequence[dict[str, str]]
    chat_template_kwargs: dict[str, object] | None


class Completion(Protocol):
    """
    Describe one generated completion returned by vLLM.
    """

    finish_reason: str | None
    text: str
    token_ids: Sequence[int]


class RequestOutput(Protocol):
    """
    Describe the completion list returned for one request.
    """

    outputs: Sequence[Completion]


class Tokenizer(Protocol):
    """
    Describe the tokenizer operation used by Harmony parsing.
    """

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


class ReasoningParser(Protocol):
    """
    Describe the reasoning parser operations used by the adapter.
    """

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
        request: ChatRequest,
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


class ReasoningParserFactory(Protocol):
    """
    Construct one reasoning parser for each completion.
    """

    def __call__(
        self,
        *,
        tokenizer: Tokenizer,
        chat_template_kwargs: dict[str, object] | None,
    ) -> ReasoningParser:
        """
        Build a parser with request-specific context.

        Args:
            tokenizer: Tokenizer associated with the generation engine.
            chat_template_kwargs: Request-specific chat template arguments, or None.

        Returns:
            A reasoning parser configured for the request.
        """
        ...


class ParserManager(Protocol):
    """
    Resolve the parser registered in the vLLM configuration.
    """

    @staticmethod
    def get_reasoning_parser(
        name: str,
    ) -> ReasoningParserFactory:
        """
        Return the registered reasoning parser factory.

        Args:
            name: Parser name registered in the engine configuration.

        Returns:
            The factory for the named reasoning parser.
        """
        ...


class StructuredOutputsConfiguration(Protocol):
    """
    Describe the configured reasoning parser name.
    """

    reasoning_parser: str | None


class VllmConfiguration(Protocol):
    """
    Describe the vLLM configuration fields used by the adapter.
    """

    structured_outputs_config: StructuredOutputsConfiguration


class Engine(Protocol):
    """
    Describe the vLLM engine configuration boundary.
    """

    vllm_config: VllmConfiguration


class VllmLanguageModel(Protocol):
    """
    Describe the offline vLLM operations used by the adapter.
    """

    llm_engine: Engine

    def get_tokenizer(
        self,
    ) -> Tokenizer:
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
    ) -> Sequence[RequestOutput]:
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


class Grammar(Protocol):
    """
    Describe XGrammar's JSON schema parser.
    """

    @staticmethod
    def from_json_schema(
        schema: dict[str, object],
    ) -> object:
        """
        Convert a response schema into a grammar.

        Args:
            schema: JSON schema submitted to vLLM.

        Returns:
            The compiled grammar.
        """
        ...


class XGrammarModule(Protocol):
    """
    Expose the grammar class from the optional vLLM dependency.
    """

    Grammar: type[Grammar]
