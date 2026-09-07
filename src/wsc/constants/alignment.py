"""
Semantic alignment defaults and relation instructions.
"""

from pathlib import Path

from ..reading.instructions import read_instructions

ALIGNMENT_FIELDS = (
    "alignment_id",
    "source_id",
    "target_id",
    "relation",
    "score",
    "context",
)
"""Columns preserving alignment evidence and inference metadata in tabular exports."""

ALIGNMENT_MODEL = "Qwen/Qwen3-Reranker-8B"
"""Default cross-encoder used to score semantic relations."""

ALIGNMENT_MODELS = (
    ALIGNMENT_MODEL,
    "Qwen/Qwen3-Reranker-4B",
    "zeroentropy/zerank-2-reranker",
    "cross-encoder/ettin-reranker-1b-v1",
)
"""Cross-encoders compared on the shared annotation sample."""

ALIGNMENT_BATCH_SIZE = 32
"""Number of definition pairs scored per inference batch."""

ALIGNMENT_MAXIMUM_LENGTH = 2048
"""Maximum token count for each encoded definition pair."""

INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.toml")
"""Bundled profiles available to commands and experiments."""

DEFAULT_INSTRUCTIONS = read_instructions(INSTRUCTIONS_PATH)[0]
"""Baseline profile used unless another profile is selected."""

ALIGNMENT_INSTRUCTION = DEFAULT_INSTRUCTIONS.instruction
"""Default general instruction, defined in the bundled TOML."""

TRANSLATION_RELATION = "translation"
"""Relation identifying the translation gloss associated with a sense."""

RELATION_INSTRUCTIONS = DEFAULT_INSTRUCTIONS.relations
"""Default relation hypotheses, defined in the bundled TOML."""
