"""Exercise SDK requests through the public alignment API."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import httpx2
import pytest
from openai import APIStatusError, OpenAI

from wsc.alignment import align_query, client
from wsc.models.alignment import ModelSettings

from .test_aligner import decision, query


@dataclass
class Server:
    """Retain mocked server responses and captured HTTP requests."""

    content: str | None = '{"s1": null, "s2": null}'
    finish_reason: str = "stop"
    status: int = 200
    requests: list[httpx2.Request] = field(default_factory=list)


@pytest.fixture
def server(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Server]:
    """
    Attach a local transport to the real OpenAI SDK.

    Args:
        monkeypatch: Test-local client replacement.

    Yields:
        Mutable responses and captured requests.
    """
    state = Server()

    def respond(
        request: httpx2.Request,
    ) -> httpx2.Response:
        """
        Capture the request and return the configured completion.

        Args:
            request: Serialized SDK request.

        Returns:
            Configured completion or error response.
        """
        state.requests.append(request)

        payload = {
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "served-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": state.finish_reason,
                    "message": {"role": "assistant", "content": state.content},
                },
            ],
        }

        return httpx2.Response(
            state.status,
            json=payload if state.status == 200 else {"error": {"message": "Rejected"}},
        )

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as transport:

        def connect(
            *,
            api_key: str,
            base_url: str,
            timeout: float,
            max_retries: int,
        ) -> OpenAI:
            """
            Construct the SDK client with the test transport.

            Args:
                api_key: Configured authentication token.
                base_url: Configured server endpoint.
                timeout: Request timeout in seconds.
                max_retries: SDK retry limit.

            Returns:
                Real SDK client with an isolated HTTP transport.
            """
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=transport,
            )

        monkeypatch.setattr(client, "OpenAI", connect)

        yield state


@pytest.mark.parametrize("effort", [None, "none", "low", "high"])
def test_sdk_preserves_generation_options_and_structured_output(
    server: Server,
    monkeypatch: pytest.MonkeyPatch,
    effort: str | None,
) -> None:
    """The real SDK serializes temperature, effort, schemas, and authentication."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    server.content = json.dumps({"s1": decision("t1"), "s2": None})
    settings = ModelSettings("served-model", reasoning_effort=effort)

    result = align_query(query(), client.open_model(settings))
    request = server.requests[0]
    payload = cast(dict[str, object], json.loads(request.content))

    assert len(server.requests) == 1
    assert str(request.url) == "http://localhost:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert payload["temperature"] == 0.0
    assert payload["max_completion_tokens"] == 4096
    assert payload["model"] == "served-model"
    assert payload["response_format"]
    assert payload.get("reasoning_effort") == effort
    assert ("reasoning_effort" in payload) == (effort is not None)
    assert result.response == server.content
    assert result.links[0].reason == "The definitions express the same concept."


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_model_outputs_remain_failures(
    server: Server,
    finish_reason: str,
) -> None:
    """Incomplete responses raise an explicit generation error."""
    server.finish_reason = finish_reason

    with pytest.raises(ValueError, match="Incomplete model response"):
        _ = align_query(query(), client.open_model(ModelSettings("served-model")))


def test_hosted_endpoint_uses_the_same_chat_protocol(
    server: Server,
) -> None:
    """An explicit hosted endpoint receives the selected model and prompt."""
    settings = ModelSettings("hosted-model", url="https://api.openai.com/v1")
    result = align_query(query(), client.open_model(settings))

    assert str(server.requests[0].url) == "https://api.openai.com/v1/chat/completions"
    assert not result.links
    assert len(result.decisions) == 2


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_errors_preserve_provider_failures(
    server: Server,
    status: int,
) -> None:
    """Rejected requests propagate SDK errors after one attempt."""
    server.status = status

    with pytest.raises(APIStatusError, match="Rejected"):
        _ = align_query(query(), client.open_model(ModelSettings("served-model")))

    assert len(server.requests) == 1


def test_refusals_remain_generation_failures(
    server: Server,
) -> None:
    """A missing completion remains distinct from a lexical null decision."""
    server.content = None

    with pytest.raises(ValueError, match="Incomplete model response"):
        _ = align_query(query(), client.open_model(ModelSettings("served-model")))
