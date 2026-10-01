"""
Data structures passed around the pipeline.
"""

from .alignment import (
    AlignmentDecision,
    AlignmentLink,
    AlignmentPrompts,
    AlignmentQuery,
    AlignmentResult,
    AlignmentTask,
    Definition,
    GlossMode,
    LanguageModel,
    ModelRequest,
    ModelSettings,
)
from .engine import Engine
from .pos import POS
from .resources import (
    Attestation,
    Example,
    Language,
    Lemma,
    Offset,
    Quotation,
    Sense,
    Sentence,
    Synset,
    SynsetAlignment,
    SynsetRelation,
    SynsetResource,
    TranslationTable,
    WordOffset,
    WordOffsetSource,
)

__all__ = [
    "POS",
    "AlignmentDecision",
    "AlignmentLink",
    "AlignmentPrompts",
    "AlignmentQuery",
    "AlignmentResult",
    "AlignmentTask",
    "Attestation",
    "Definition",
    "Engine",
    "Example",
    "GlossMode",
    "Language",
    "LanguageModel",
    "Lemma",
    "ModelRequest",
    "ModelSettings",
    "Offset",
    "Quotation",
    "Sense",
    "Sentence",
    "Synset",
    "SynsetAlignment",
    "SynsetRelation",
    "SynsetResource",
    "TranslationTable",
    "WordOffset",
    "WordOffsetSource",
]
