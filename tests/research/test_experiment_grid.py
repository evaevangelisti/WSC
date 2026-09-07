"""Exercise experiment grids and cache replay without downloading model weights."""

import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pytest
from experiments.scripts import run

from wsc.constants import ALIGNMENT_MODELS, INSTRUCTIONS_PATH
from wsc.models import POS
from wsc.models.alignment import (
    AlignmentInstructions,
    AlignmentQuery,
    AlignmentTask,
    Comparison,
    Definition,
    GlossMode,
    Scorer,
)
from wsc.reading import read_alignments, read_instructions, read_metadata


@pytest.mark.parametrize("task", list(AlignmentTask))
@pytest.mark.parametrize("selection", ["default", "filtered"])
def test_grid_scores_every_combination_and_replays_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    task: AlignmentTask,
    selection: str,
) -> None:
    """Every experimental combination survives cache replay for both resources."""
    query = AlignmentQuery(
        task,
        "entry",
        "entry",
        "word",
        POS.NOUN,
        (Definition("source", ("ancestor marker", "leaf marker")),),
        (Definition("target", ("candidate marker",)),),
    )
    sample = tmp_path / "sample.json"
    _ = sample.write_text(
        json.dumps(
            {
                "seed": 0,
                "queries": [asdict(query)],
                "splits": {query.alignment_id: "development"},
            }
        ),
        encoding="utf-8",
    )
    profiles = read_instructions(INSTRUCTIONS_PATH)
    models = ALIGNMENT_MODELS
    modes = tuple(GlossMode)
    arguments = [
        "experiments",
        str(sample),
        "--device",
        "cpu",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if selection == "filtered":
        profiles = profiles[::2]
        models = models[::2]
        modes = modes[:2]
        for option, values in (
            ("--model", models),
            ("--gloss-mode", modes),
            ("--instruction-profile", tuple(profile.name for profile in profiles)),
        ):
            for value in values:
                arguments.extend((option, value))
    loaded: list[tuple[str, str]] = []
    received: list[Comparison] = []

    def load_model(
        model: str,
        *,
        instructions: AlignmentInstructions,
        **options: object,
    ) -> Scorer:
        """
        Replace model loading while checking profile propagation.

        Args:
            model: Requested cross-encoder identifier.
            instructions: Selected general instruction and relation hypotheses.
            options: Inference settings forwarded by the experiment command.

        Returns:
            An inference boundary recording every comparison.
        """
        assert options["device"] == "cpu"
        assert instructions in profiles
        loaded.append((model, instructions.name))

        class Model:
            """Record semantic inputs without interpreting their content."""

            def score(self, pairs: Sequence[Comparison]) -> list[float]:
                """
                Verify relation instructions and retain model inputs.

                Args:
                    pairs: Complete definition comparisons for one query.

                Returns:
                    One finite score per candidate relation.
                """
                assert all(
                    any(text in pair.query for text in instructions.relations.values())
                    for pair in pairs
                )
                received.extend(pairs)

                return [1.0] * len(pairs)

        return Model()

    monkeypatch.setattr(run, "CrossEncoderScorer", load_model)
    monkeypatch.setattr(sys, "argv", arguments)
    run.main()
    paths = [Path(line) for line in capsys.readouterr().out.splitlines()]
    expected = set(product(models, modes, (profile.name for profile in profiles)))
    metadata = [read_metadata(path) for path in paths]

    assert len(paths) == len(set(paths)) == len(expected)
    assert {
        (record["model"], record["gloss_mode"], record["instruction_profile"])
        for record in metadata
    } == expected
    assert loaded == list(product(models, (profile.name for profile in profiles)))
    for path in paths:
        (result,) = read_alignments(path)
        assert result.query == query
        assert len(result.scores) == (1 if task == AlignmentTask.TRANSLATIONS else 3)
    assert any("Wiktionary: leaf marker\n" in pair.query for pair in received)
    assert any(
        "Wiktionary: ancestor marker leaf marker\n" in pair.query for pair in received
    )
    assert any("Ancestor context:" in pair.query for pair in received) == (
        GlossMode.CONTEXT in modes
    )
    evidence = {path: path.read_bytes() for path in paths}
    loaded.clear()
    received.clear()
    monkeypatch.setattr(sys, "argv", [*arguments, "--reuse"])
    run.main()

    assert [Path(line) for line in capsys.readouterr().out.splitlines()] == paths
    assert not loaded
    assert not received
    assert {path: path.read_bytes() for path in paths} == evidence
