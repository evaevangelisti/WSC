"""Exercise offline vLLM inference without loading model weights."""

import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import ClassVar, override

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import align_query
from wsc.models.alignment import ModelSettings

from .examples import build_decision, build_query


class FakeSamplingParameters:
    """Capture generation settings passed to vLLM."""

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        """Store the sampling settings."""
        self.values: dict[str, object] = kwargs


class FakeTokenizer:
    """Decode fake completion token identifiers."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Return the configured final response."""
        assert skip_special_tokens

        return "".join(chr(token) for token in token_ids if token >= 0)


@dataclass
class FakeChatRequest:
    """Carry the request fields used by offline parsing and sampling."""

    model: str
    messages: list[dict[str, str]]
    chat_template_kwargs: dict[str, object] | None


class FakeParser:
    """Separate reasoning and final content at a synthetic boundary token."""

    instances: ClassVar[list[FakeParser]] = []

    def __init__(
        self,
        *,
        tokenizer: FakeTokenizer,
        **kwargs: object,
    ) -> None:
        """Store parser context without loading a tokenizer vocabulary."""
        self.tokenizer: FakeTokenizer = tokenizer
        self.options: dict[str, object] = kwargs

        self.instances.append(self)

    def extract_reasoning(
        self,
        model_output: str,
        request: FakeChatRequest,
    ) -> tuple[str | None, str | None]:
        """Use template context to distinguish final text from reasoning."""
        assert request.chat_template_kwargs == self.options["chat_template_kwargs"]

        options = request.chat_template_kwargs or {}

        if options.get("enable_thinking") is False:
            return None, model_output

        reasoning, boundary, content = model_output.partition("</think>")

        return reasoning, content if boundary else None


class FakeHarmonyParser(FakeParser):
    """Extract Harmony content exclusively from generated token identifiers."""

    def extract_content_ids(
        self,
        input_ids: list[int],
    ) -> list[int]:
        """Separate Harmony content at a synthetic channel boundary."""
        if -1 not in input_ids:
            return []

        return input_ids[input_ids.index(-1) + 1 :]

    @override
    def extract_reasoning(
        self,
        model_output: str,
        request: FakeChatRequest,
    ) -> tuple[str | None, str | None]:
        """Reject text extraction as the vLLM 0.17.0 GPT-OSS parser does."""
        del model_output, request

        raise NotImplementedError


class FakeCustomHarmonyParser(FakeHarmonyParser):
    """Represent a custom GPT-OSS parser registered under another name."""


class FakeParserManager:
    """Resolve the parser registered for an explicit reasoning format."""

    selections: ClassVar[list[str]] = []

    @classmethod
    def get_reasoning_parser(
        cls,
        name: str,
    ) -> type[FakeParser]:
        """Return the parser selected in the engine configuration."""
        cls.selections.append(name)

        if name == "openai_gptoss":
            return FakeHarmonyParser

        if name == "custom_gptoss":
            return FakeCustomHarmonyParser

        return FakeParser


class FakeLanguageModel:
    """Return one configured vLLM completion without loading a model."""

    response: str = json.dumps({"s1": build_decision("t1"), "s2": None})
    finish_reason: str | None = "stop"
    token_ids: ClassVar[list[int]] = []
    instances: ClassVar[list[FakeLanguageModel]] = []

    def __init__(
        self,
        **kwargs: object,
    ) -> None:
        """Store engine settings without loading model weights."""
        self.settings: dict[str, object] = kwargs
        self.llm_engine: SimpleNamespace = SimpleNamespace(
            vllm_config=SimpleNamespace(
                structured_outputs_config=SimpleNamespace(
                    reasoning_parser=kwargs.get("reasoning_parser", ""),
                ),
            ),
        )
        self.requests: list[dict[str, object]] = []

        self.__class__.instances.append(self)

    def get_tokenizer(
        self,
    ) -> FakeTokenizer:
        """Return the fake tokenizer used by the reasoning parser."""
        return FakeTokenizer()

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        sampling_params: FakeSamplingParameters,
        chat_template_kwargs: dict[str, object] | None,
        use_tqdm: bool,
    ) -> list[object]:
        """Return the configured completion and retain the request."""
        assert not use_tqdm

        self.requests.append(
            {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template_kwargs": chat_template_kwargs,
            },
        )

        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        finish_reason=self.finish_reason,
                        text=self.response,
                        token_ids=self.token_ids,
                    ),
                ],
            ),
        ]


@pytest.fixture(name="_vllm_modules")
def fake_vllm_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide a fake vLLM package for the lazy provider imports."""
    FakeLanguageModel.instances.clear()
    FakeParser.instances.clear()
    FakeParserManager.selections.clear()

    FakeLanguageModel.finish_reason = "stop"
    FakeLanguageModel.response = json.dumps({"s1": build_decision("t1"), "s2": None})
    FakeLanguageModel.token_ids = []

    vllm = ModuleType("vllm")
    monkeypatch.setattr(vllm, "LLM", FakeLanguageModel, raising=False)
    monkeypatch.setattr(vllm, "SamplingParams", FakeSamplingParameters, raising=False)

    parser = ModuleType("vllm.reasoning")
    monkeypatch.setattr(
        parser,
        "ReasoningParserManager",
        FakeParserManager,
        raising=False,
    )

    sampling_params = ModuleType("vllm.sampling_params")
    monkeypatch.setattr(
        sampling_params,
        "SamplingParams",
        FakeSamplingParameters,
        raising=False,
    )
    monkeypatch.setattr(
        sampling_params,
        "StructuredOutputsParams",
        FakeSamplingParameters,
        raising=False,
    )

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.reasoning", parser)

    harmony = ModuleType("vllm.reasoning.gptoss_reasoning_parser")
    monkeypatch.setattr(
        harmony,
        "GptOssReasoningParser",
        FakeHarmonyParser,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, harmony.__name__, harmony)

    protocol = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    monkeypatch.setattr(
        protocol,
        "ChatCompletionRequest",
        FakeChatRequest,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)


@pytest.mark.usefixtures("_vllm_modules")
def test_forwards_generation_settings() -> None:
    """The vLLM adapter forwards settings and returns validated JSON."""
    from wsc.alignment.inference import open_model

    settings = ModelSettings(
        "local-model",
        reasoning_effort="low",
        engine_options=(("dtype", "float16"),),
        chat_template_options=(("enable_thinking", False),),
    )
    result = align_query(build_query(), open_model(settings))

    model = FakeLanguageModel.instances[0]
    request = model.requests[0]
    sampling = request["sampling_params"]

    assert model.settings == {"model": "local-model", "dtype": "float16"}
    assert isinstance(sampling, FakeSamplingParameters)
    assert sampling.values["temperature"] == 0.0
    assert sampling.values["max_tokens"] == 4096

    structured_outputs = sampling.values["structured_outputs"]

    assert isinstance(structured_outputs, FakeSamplingParameters)

    schema = structured_outputs.values["json"]

    assert isinstance(schema, dict)
    assert schema["required"] == ["s1", "s2"]
    assert request["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "enable_thinking": False,
    }
    assert result.response == FakeLanguageModel.response
    assert result.links[0].reason == "The definitions express the same concept."


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
@pytest.mark.usefixtures("_vllm_modules")
def test_rejects_incomplete_output(
    finish_reason: str | None,
) -> None:
    """Incomplete offline completions raise an explicit generation error."""
    from wsc.alignment.inference import open_model

    FakeLanguageModel.finish_reason = finish_reason

    with pytest.raises(ValueError, match="Incomplete model response"):
        _ = align_query(build_query(), open_model(ModelSettings("local-model")))


def test_reports_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the package remains possible when the optional backend is absent."""
    monkeypatch.setitem(sys.modules, "vllm", None)

    from wsc.alignment.inference import open_model

    with pytest.raises(RuntimeError, match="requires vLLM"):
        _ = open_model(ModelSettings("local-model"))


@pytest.mark.parametrize(
    "parser",
    ["openai_gptoss", "custom_gptoss", "qwen3", "deepseek_r1"],
)
@given(reason=st.text(min_size=1).filter(lambda text: bool(text.strip())))
@pytest.mark.usefixtures("_vllm_modules")
def test_uses_selected_parser(
    parser: str,
    reason: str,
) -> None:
    """Only final tokens reach JSON validation, even when reasoning contains JSON."""
    from wsc.alignment.inference import open_model

    response = json.dumps(
        {
            "s1": [{"target_id": "t1", "relation": "translation", "reason": reason}],
            "s2": None,
        },
    )
    FakeLanguageModel.response = '{"s1": null, "s2": null} reasoning</think>' + response
    FakeLanguageModel.token_ids = [
        *map(ord, FakeLanguageModel.response),
        -1,
        *map(ord, response),
        -2,
    ]
    model = open_model(ModelSettings("local-model", reasoning_parser=parser))

    result = align_query(build_query(), model)

    assert result.response == response
    assert result.links[0].target_id == "t1"
    assert FakeLanguageModel.instances[-1].settings["reasoning_parser"] == parser

    sampling = FakeLanguageModel.instances[-1].requests[0]["sampling_params"]

    assert isinstance(sampling, FakeSamplingParameters)

    structured = sampling.values["structured_outputs"]

    assert isinstance(structured, FakeSamplingParameters)
    assert "json" in structured.values
    assert sampling.values["skip_special_tokens"] is False


@pytest.mark.parametrize("parser", [None, "qwen3", "openai_gptoss"])
@pytest.mark.usefixtures("_vllm_modules")
def test_rejects_empty_responses(
    parser: str | None,
) -> None:
    """A stopped generation still requires nonempty final content."""
    from wsc.alignment.inference import open_model

    FakeLanguageModel.response = "reasoning</think> \n\t" if parser else " \n\t"
    FakeLanguageModel.token_ids = [*map(ord, "reasoning"), -1, *map(ord, " \n\t")]

    with pytest.raises(ValueError, match="Empty final model response"):
        _ = align_query(
            build_query(),
            open_model(ModelSettings("local-model", reasoning_parser=parser)),
        )


@pytest.mark.parametrize("parser", ["qwen3", "openai_gptoss"])
@pytest.mark.usefixtures("_vllm_modules")
def test_requires_final_channel(
    parser: str,
) -> None:
    """Reasoning JSON cannot be accepted as the final decision."""
    from wsc.alignment.inference import open_model

    FakeLanguageModel.token_ids = list(map(ord, FakeLanguageModel.response))

    with pytest.raises(ValueError, match="Empty final model response"):
        _ = align_query(
            build_query(),
            open_model(ModelSettings("local-model", reasoning_parser=parser)),
        )


@pytest.mark.usefixtures("_vllm_modules")
def test_preserves_disabled_thinking() -> None:
    """Template options reach parsing when a reasoning model generates plain JSON."""
    from wsc.alignment.inference import open_model

    result = align_query(
        build_query(),
        open_model(
            ModelSettings(
                "local-model",
                reasoning_parser="qwen3",
                chat_template_options=(("enable_thinking", False),),
            ),
        ),
    )

    assert result.response == FakeLanguageModel.response
    assert result.links[0].target_id == "t1"


@pytest.mark.parametrize("parser", [None, "qwen3"])
@pytest.mark.usefixtures("_vllm_modules")
def test_prioritizes_explicit_parser(
    parser: str | None,
) -> None:
    """Generation and extraction use the same parser after option precedence."""
    from wsc.alignment.inference import open_model

    response = FakeLanguageModel.response
    FakeLanguageModel.response = "reasoning</think>" + response
    result = align_query(
        build_query(),
        open_model(
            ModelSettings(
                "local-model",
                reasoning_parser=parser,
                engine_options=(("reasoning_parser", "deepseek_r1"),),
            ),
        ),
    )

    assert result.response == response
    assert FakeParserManager.selections == [parser or "deepseek_r1"]
    assert FakeLanguageModel.instances[0].settings["reasoning_parser"] == (
        parser or "deepseek_r1"
    )


@pytest.mark.usefixtures("_vllm_modules")
def test_isolates_harmony_parsers() -> None:
    """Repeated generations parse independent final channels for local checkpoints."""
    from wsc.alignment.inference import open_model

    response = FakeLanguageModel.response
    FakeLanguageModel.token_ids = [-1, *map(ord, response), -2]
    model = open_model(
        ModelSettings("/models/custom-checkpoint", reasoning_parser="openai_gptoss"),
    )

    first = align_query(build_query(), model)
    second = align_query(build_query(), model)

    assert first == second
    assert first.response == response
    assert FakeParserManager.selections == ["openai_gptoss"]
    assert len(FakeParser.instances) == 2
    assert FakeParser.instances[0] is not FakeParser.instances[1]
