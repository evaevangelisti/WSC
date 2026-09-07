"""
Data structures passed around the pipeline.
"""

from .alignment import (
    AlignmentInstructions,
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Comparison,
    Definition,
    GlossMode,
    Reranker,
    Scorer,
)
from .engine import Engine
from .pos import POS
from .resources import (
    Attestation,
    Example,
    Lemma,
    Offset,
    Quotation,
    Sense,
    Sentence,
    Synset,
    Translations,
    WordNetAlignment,
    WordNetRelation,
    WordOffset,
    WordOffsetSource,
)

__all__ = [
    "POS",
    "AlignmentInstructions",
    "AlignmentQuery",
    "AlignmentResult",
    "AlignmentScore",
    "AlignmentTask",
    "Attestation",
    "Comparison",
    "Definition",
    "Engine",
    "Example",
    "GlossMode",
    "Lemma",
    "Offset",
    "Quotation",
    "Reranker",
    "Scorer",
    "Sense",
    "Sentence",
    "Synset",
    "Translations",
    "WordNetAlignment",
    "WordNetRelation",
    "WordOffset",
    "WordOffsetSource",
]
