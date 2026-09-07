"""Generate decisions through OpenAI-compatible chat completion endpoints."""

import os
from typing import cast

from openai import OpenAI, omit
from openai.types.shared import ReasoningEffort

from ..constants import ALIGNMENT_TIMEOUT
from ..models.alignment import LanguageModel, ModelRequest, ModelSettings


class ChatModel:
    """Use the OpenAI chat protocol for vLLM and hosted models."""

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        """
        Configure the model endpoint.

        Args:
            settings: Endpoint and generation configuration.
        """
        self._client: OpenAI = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=settings.url,
            timeout=ALIGNMENT_TIMEOUT,
            max_retries=0,
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
            APIStatusError: If the endpoint rejects the request.
            ValueError: If generation is incomplete or has no text.
        """
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.maximum_tokens,
            reasoning_effort=(
                cast(ReasoningEffort, self._settings.reasoning_effort)
                if self._settings.reasoning_effort is not None
                else omit
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "alignment",
                    "strict": True,
                    "schema": request.schema,
                },
            },
        )

        choice = response.choices[0]

        if choice.finish_reason != "stop" or choice.message.content is None:
            raise ValueError(f"Incomplete model response: {choice.finish_reason}")

        return choice.message.content


def open_model(
    settings: ModelSettings,
) -> LanguageModel:
    """
    Construct a client for the configured language model server.

    Args:
        settings: Model identifier, endpoint, and generation options.

    Returns:
        An OpenAI-compatible chat client.
    """
    return ChatModel(settings)
