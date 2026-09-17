"""Define language model defaults and alignment record fields."""

from pathlib import Path

from ..reading.prompts import read_prompts

ALIGNMENT_DIR = Path("alignment")
"""Default destination for aligned senses and their reports."""

ALIGNMENT_FIELDS = (
    "alignment_id",
    "source_id",
    "target_id",
    "relation",
    "reason",
)
"""Store one tabular row for each source decision."""

ALIGNMENT_MANIFEST = "manifest.json"
"""Name the provenance document written beside aligned senses."""

ALIGNMENT_REPORTS_DIR = "reports"
"""Name the directory containing one report per alignment task."""

ALIGNMENT_SENSES = "senses.jsonl"
"""Name the aligned collection stored in an output directory."""

ALIGNMENT_MODEL = "openai/gpt-oss-120b"
"""Select the default local inference model."""

ALIGNMENT_MAXIMUM_TOKENS = 4096
"""Limit the generated response length."""

ALIGNMENT_TEMPERATURE = 0.0
"""Request zero-temperature generation."""

ALIGNMENT_BATCH_SIZE = 32768
"""Build this many prompts before each inference pass."""

PROMPTS_PATH = Path(__file__).with_name("prompts.toml")
"""Locate the bundled task prompt templates."""

DEFAULT_PROMPTS = read_prompts(PROMPTS_PATH)
"""Load one default prompt template per task."""

TRANSLATION_RELATION = "translation"
"""Identify a translation association."""

HIERARCHY_CONSTRAINT = (
    "- Read each '>' hierarchy from left to right, interpreting the final sense "
    + "with its inherited defining restrictions."
)
"""Explain the full Wiktionary gloss representation."""
