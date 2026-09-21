"""Expose alignment statistics, provenance, and output publication."""

from .manifest import build_manifest
from .statistics import AlignmentStatistics
from .writing import write_alignment

__all__ = [
    "AlignmentStatistics",
    "build_manifest",
    "write_alignment",
]
