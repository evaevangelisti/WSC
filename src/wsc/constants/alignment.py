"""Define language model defaults and alignment record fields."""

from pathlib import Path

from ..reading.prompts import read_prompts

ALIGNMENT_FIELDS = (
    "alignment_id",
    "source_id",
    "target_id",
    "relation",
    "reason",
)
"""Store one tabular row for each source decision."""

ALIGNMENT_MODEL = "openai/gpt-oss-120b"
"""Select the default local inference model."""

ALIGNMENT_MAXIMUM_TOKENS = 4096
"""Limit the generated response length."""

ALIGNMENT_TEMPERATURE = 0.0
"""Request zero-temperature generation."""

PROMPTS_PATH = Path(__file__).with_name("prompts.toml")
"""Locate the bundled task prompt templates."""

DEFAULT_PROMPTS = read_prompts(PROMPTS_PATH)
"""Load one default prompt template per task."""

TRANSLATION_RELATION = "translation"
"""Identify a translation association."""

ALIGNMENT_SCHEMA = "5"
"""Version the decision-based cache independently of reranker scores."""

HIERARCHY_CONSTRAINT = (
    "- Read each '>' hierarchy from general context to the final specific sense."
)
"""Explain the full Wiktionary gloss representation."""
