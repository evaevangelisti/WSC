"""
Language selection and extraction defaults.
"""

LANGUAGE = "en"
"""Wiktionary's code for the edition read, and for the language kept in it."""

LANGUAGE_SECTION = "English"
"""Language heading used to locate English entries in Wiktionary pages."""

COMPRESSION_LEVEL = 10
"""How hard a parsed dump is packed, written once and read whole every time."""

SPACY_PIPELINE = "en_core_web_trf"
"""Transformer pipeline selected for accurate sentence lemmatization."""

BATCH_SIZE = 8
"""Small extraction batches limit padding when quotation lengths vary."""

PROCESSES = 1
"""Single-process extraction avoids contention with Torch's own parallelism."""
