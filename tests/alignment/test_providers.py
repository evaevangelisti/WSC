"""Exercise the offline vLLM model boundary without loading model weights."""

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from wsc.alignment import align_query
from wsc.models.alignment import ModelSettings

from .test_aligner import decision, query


class FakeSamplingParams:
    """Capture generation settings passed to vLLM."""

    def __init__(self, **kwargs: object) -> None:
        """Store the sampling settings."""
        self.values = kwargs


class FakeStructuredOutputsParams:
    """Capture the structured-output schema passed to vLLM."""

    def __init__(self, **kwargs: object) -> None:
        """Store the guided decoding settings."""
        self.values = kwargs


class FakeTokenizer:
    """Decode fake completion token identifiers."""

    def decode(self, token_ids: list[int]) -> str:
        """Return the configured final response."""
        del token_ids
        return FakeLLM.response


class FakeReasoningParser:
    """Extract the final content from fake GPT-OSS output tokens."""

    def __init__(self, *, tokenizer: FakeTokenizer) -> None:
        """Store the tokenizer used by the parser."""
        self.tokenizer = tokenizer

    def extract_content_ids(self, token_ids: list[int]) -> list[int]:
        """Return token identifiers for the final response."""
        return token_ids


class FakeReasoningParserManager:
    """Resolve the fake GPT-OSS reasoning parser."""

    @staticmethod
    def get_reasoning_parser(name: str) -> type[FakeReasoningParser]:
        """Return the parser for the configured model format."""
        assert name == "openai_gptoss"
        return FakeReasoningParser


class FakeLLM:
    """Return one configured vLLM completion without loading a model."""

    response = json.dumps({"s1": decision("t1"), "s2": None})
    finish_reason = "stop"
    instances: ClassVar[list[FakeLLM]] = []

    def __init__(self, **kwargs: object) -> None:
        """Store engine settings without loading model weights."""
        self.settings = kwargs
        self.requests: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def get_tokenizer(self) -> FakeTokenizer:
        """Return the fake tokenizer used by the reasoning parser."""
        return FakeTokenizer()

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        sampling_params: FakeSamplingParams,
        chat_template_kwargs: dict[str, str] | None,
    ) -> list[object]:
        """Return the configured completion and retain the request."""
        self.requests.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason=self.finish_reason,
                        text=self.response,
                        token_ids=[1, 2, 3],
                    )
                ]
            )
        ]


@pytest.fixture(name="_vllm_modules")
def fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake vLLM package for the lazy provider imports."""
    FakeLLM.instances.clear()
    FakeLLM.finish_reason = "stop"

    vllm = ModuleType("vllm")
    vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]

    reasoning = ModuleType("vllm.reasoning")
    reasoning.ReasoningParserManager = (  # type: ignore[attr-defined]
        FakeReasoningParserManager
    )

    sampling_params = ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = (  # type: ignore[attr-defined]
        FakeStructuredOutputsParams
    )

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.reasoning", reasoning)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)


@pytest.mark.usefixtures("_vllm_modules")
def test_offline_model_preserves_generation_contract() -> None:
    """The vLLM adapter forwards settings and returns validated JSON."""
    from wsc.alignment.client import open_model

    settings = ModelSettings(
        "local-model",
        reasoning_effort="low",
        engine_options=(("dtype", "float16"),),
    )
    result = align_query(query(), open_model(settings))

    model = FakeLLM.instances[0]
    request = model.requests[0]
    sampling = request["sampling_params"]
    assert model.settings == {"model": "local-model", "dtype": "float16"}
    assert isinstance(sampling, FakeSamplingParams)
    assert sampling.values["temperature"] == 0.0
    assert sampling.values["max_tokens"] == 4096
    structured_outputs = sampling.values["structured_outputs"]
    assert isinstance(structured_outputs, FakeStructuredOutputsParams)
    schema = structured_outputs.values["json"]
    assert isinstance(schema, dict)
    assert schema["required"] == ["s1", "s2"]
    assert request["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert result.response == FakeLLM.response
    assert result.links[0].reason == "The definitions express the same concept."


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
@pytest.mark.usefixtures("_vllm_modules")
def test_incomplete_model_outputs_remain_failures(finish_reason: str) -> None:
    """Incomplete offline completions raise an explicit generation error."""
    from wsc.alignment.client import open_model

    FakeLLM.finish_reason = finish_reason

    with pytest.raises(ValueError, match="Incomplete model response"):
        _ = align_query(query(), open_model(ModelSettings("local-model")))


def test_missing_vllm_is_reported_at_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the package remains possible when the optional backend is absent."""
    monkeypatch.setitem(sys.modules, "vllm", None)

    from wsc.alignment.client import open_model

    with pytest.raises(RuntimeError, match="requires vLLM"):
        _ = open_model(ModelSettings("local-model"))
