"""
Inference provenance and compatibility fingerprints.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from ..constants import DEFAULT_INSTRUCTIONS
from ..models.alignment import AlignmentInstructions


def build_metadata(
    input_path: Path,
    model: str,
    revision: str,
    mode: str,
    maximum_length: int,
    synsets_path: Path | None = None,
    instructions: AlignmentInstructions = DEFAULT_INSTRUCTIONS,
) -> dict[str, str]:
    """
    Record settings that affect semantic evidence.

    Args:
        input_path: Collection or frozen research sample.
        model: Cross-encoder identifier.
        revision: Requested model revision.
        mode: Source gloss representation.
        maximum_length: Token truncation boundary.
        synsets_path: Cached WordNet source, when used.
        instructions: Complete instruction profile used during inference.

    Returns:
        Content fingerprints and inference settings, excluding decision thresholds.
    """
    instruction_text = json.dumps(
        asdict(instructions),
        sort_keys=True,
    )

    return {
        "schema": "3",
        "input": fingerprint(input_path),
        "model": model,
        "revision": revision,
        "gloss_mode": mode,
        "maximum_length": str(maximum_length),
        "instruction_profile": instructions.name,
        "instruction_text": instruction_text,
        "instructions": hashlib.sha256(instruction_text.encode()).hexdigest(),
        "wordnet": fingerprint(synsets_path) if synsets_path else "",
    }


def fingerprint(
    path: Path,
) -> str:
    """
    Identify an input by content rather than filesystem timestamps.

    Args:
        path: Source artifact.

    Returns:
        SHA-256 digest of the file bytes.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")

    return digest.hexdigest()


def cache_key(
    metadata: Mapping[str, str],
) -> str:
    """
    Identify compatible evidence independently of assignment thresholds.

    Args:
        metadata: Model, input, resource, and representation settings.

    Returns:
        Stable cache directory name.
    """
    content = json.dumps(dict(metadata), sort_keys=True).encode()

    return hashlib.sha256(content).hexdigest()
