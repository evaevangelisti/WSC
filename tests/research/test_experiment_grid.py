"""Exercise complete experiment grids, manual reports, and cached replay."""

import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import cast

import pytest
from annotation.scripts.alignment import build_task
from experiments.scripts import run

from wsc.models import POS
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentTask,
    Definition,
    GlossMode,
    LanguageModel,
    ModelRequest,
    ModelSettings,
)
from wsc.reading import read_alignments, read_metadata


@pytest.fixture
def arguments(
    tmp_path: Path,
) -> list[str]:
    """
    Prepare two task samples and their independent manual annotations.

    Args:
        tmp_path: Isolated input and output directory.

    Returns:
        Arguments for the combined experiment command.
    """
    samples: list[Path] = []
    annotations: list[dict[str, object]] = []

    for task in AlignmentTask:
        queries = [
            AlignmentQuery(
                task,
                split,
                split,
                split,
                POS.NOUN,
                (Definition("source", ("parent", "leaf"), ("synonym",)),),
                (Definition("target", ("candidate",)),),
            )
            for split in ("development", "test")
        ]

        sample = tmp_path / f"{task}.json"

        _ = sample.write_text(
            json.dumps(
                {
                    "seed": 0,
                    "queries": [asdict(query) for query in queries],
                    "splits": {
                        query.alignment_id: query.alignment_id for query in queries
                    },
                },
            ),
            encoding="utf-8",
        )

        samples.append(sample)

        for query in queries:
            annotation = build_task(query)
            annotation["annotations"] = [
                {
                    "completed_by": "annotator",
                    "result": [
                        {"from_name": "status", "value": {"choices": ["no_match"]}},
                        {
                            "from_name": "notes",
                            "value": {"text": ["Checked independently."]},
                        },
                    ],
                },
            ]
            annotations.append(annotation)

    annotation_path = tmp_path / "annotations.json"
    _ = annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    return [
        "experiments",
        *(str(path) for path in samples),
        str(tmp_path / "reports"),
        "--annotations",
        str(annotation_path),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--bootstrap",
        "5",
    ]


def test_complete_grid_preserves_settings_reports_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    """Every configuration produces distinct evidence and a task report."""
    prompt_paths = [tmp_path / "first.toml", tmp_path / "second.toml"]

    for path in prompt_paths:
        template = f"{path.stem} $source_definitions $target_definitions"

        _ = path.write_text(
            (
                f'[translations]\ntemplate = "{template}"\n'
                f'[wordnet]\ntemplate = "{template}"\n'
            ),
            encoding="utf-8",
        )

    options = [
        *arguments,
        "--model",
        "model-a",
        "--model",
        "model-b",
        "--url",
        "http://localhost:8000/v1",
        "--url",
        "http://localhost:8001/v1",
        "--prompts",
        str(prompt_paths[0]),
        "--prompts",
        str(prompt_paths[1]),
        "--temperature",
        "0",
        "--temperature",
        "0.3",
        "--reasoning-effort",
        "unset",
        "--reasoning-effort",
        "low",
    ]

    loaded: list[ModelSettings] = []
    requests: list[ModelRequest] = []

    class Model:
        """Record requests and provide valid empty decisions."""

        def generate(
            self,
            request: ModelRequest,
        ) -> str:
            """Return an explicit null decision and retain the prompt."""
            requests.append(request)

            return '{"source": null}'

    def open_model(
        settings: ModelSettings,
    ) -> LanguageModel:
        """Record inference settings and return the offline model."""
        loaded.append(settings)

        return Model()

    monkeypatch.setattr(run, "open_model", open_model)
    monkeypatch.setattr(sys, "argv", options)

    run.main()

    output = capsys.readouterr().out.splitlines()
    paths = [Path(line) for line in output if not line.startswith("Wrote ")]
    metadata = [read_metadata(path) for path in paths]

    observed: set[tuple[str, str, str, str, float, str | None]] = set()

    for path, item in zip(paths, metadata, strict=True):
        settings = cast(dict[str, object], json.loads(item["settings"]))
        temperature = cast(float, settings["temperature"])
        effort = cast(str | None, settings["reasoning_effort"])

        observed.add(
            (
                path.stem,
                item["model"],
                item["gloss_mode"],
                item["prompt"],
                temperature,
                effort,
            ),
        )

        expected_port = 8000 if item["model"] == "model-a" else 8001

        assert settings["url"] == f"http://localhost:{expected_port}/v1"
        assert len(list(read_alignments(path))) == 2

    assert len(paths) == len(set(paths)) == len(loaded) == 64
    assert len(requests) == 128
    assert observed == set(
        product(
            tuple(AlignmentTask),
            ("model-a", "model-b"),
            tuple(GlossMode),
            ("first", "second"),
            (0.0, 0.3),
            (None, "low"),
        ),
    )
    assert any(
        "source (synonym) parent > leaf" in request.prompt for request in requests
    )
    assert any("source (synonym) leaf" in request.prompt for request in requests)

    documents: dict[Path, str] = {}

    for task in AlignmentTask:
        path = tmp_path / "reports" / f"{task}.md"
        document = path.read_text(encoding="utf-8")
        documents[path] = document

        assert "Held-out configuration comparison" in document
        assert "Temperature | Reasoning effort" in document
        assert "Checked independently." in document
        assert "| 0.3 | low |" in document

    evidence = {path: path.read_bytes() for path in paths}
    requests.clear()
    loaded.clear()

    monkeypatch.setattr(sys, "argv", [*options, "--reuse"])

    run.main()

    assert capsys.readouterr().out.splitlines() == output
    assert not requests
    assert not loaded
    assert {path: path.read_bytes() for path in paths} == evidence
    assert {path: path.read_text(encoding="utf-8") for path in documents} == documents


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            [
                "--model",
                "a",
                "--model",
                "b",
                "--model",
                "c",
                "--url",
                "http://one/v1",
                "--url",
                "http://two/v1",
            ],
            "endpoint",
        ),
        (["--temperature", "-0.1"], "Temperatures"),
        (["--temperature", "2.1"], "Temperatures"),
        (["--maximum-tokens", "0"], "Maximum tokens"),
        (["--reuse"], "compatible experiment cache"),
    ],
)
def test_invalid_grid_inputs_fail_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    options: list[str],
    message: str,
) -> None:
    """Invalid configuration and absent replay evidence fail before API requests."""

    def unavailable(
        settings: ModelSettings,
    ) -> LanguageModel:
        """Reject unexpected model access."""
        pytest.fail(f"Unexpected model request: {settings}")

    monkeypatch.setattr(run, "open_model", unavailable)
    monkeypatch.setattr(sys, "argv", [*arguments, *options])

    with pytest.raises(ValueError, match=message):
        run.main()
