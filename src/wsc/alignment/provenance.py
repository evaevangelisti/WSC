"""Fingerprint model settings, prompts, and lexical inputs."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from ..constants import ALIGNMENT_SCHEMA, DEFAULT_PROMPTS
from ..models.alignment import AlignmentPrompts, GlossMode, ModelSettings

type Metadata = dict[str, object]


def build_metadata(
    input_path: Path,
    settings: ModelSettings,
    mode: GlossMode,
    synsets_path: Path | None = None,
    prompts: AlignmentPrompts = DEFAULT_PROMPTS,
) -> Metadata:
    """
    Record the configuration that produced alignment decisions.

    Args:
        input_path: Collection or frozen research sample.
        settings: Model runtime and generation configuration.
        mode: Wiktionary gloss representation.
        synsets_path: Cached WordNet source.
        prompts: Complete task prompt templates.

    Returns:
        Content fingerprints and reproducible inference settings.
    """
    prompt_text = json.dumps(asdict(prompts), sort_keys=True)

    return {
        "schema": ALIGNMENT_SCHEMA,
        "input": fingerprint(input_path),
        "model": settings.model,
        "settings": json.dumps(asdict(settings), sort_keys=True),
        "gloss_mode": mode,
        "prompts": {
            "name": prompts.name,
            "text": prompt_text,
            "fingerprint": hashlib.sha256(prompt_text.encode()).hexdigest(),
        },
        "wordnet": fingerprint(synsets_path) if synsets_path else "",
    }


def fingerprint(
    path: Path,
) -> str:
    """
    Compute a file content fingerprint.

    Args:
        path: Source artifact.

    Returns:
        SHA-256 digest of the file bytes.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")

    return digest.hexdigest()


def cache_key(
    metadata: Mapping[str, object],
) -> str:
    """
    Identify a model run by its configuration and inputs.

    Args:
        metadata: Model settings, prompts, and source fingerprints.

    Returns:
        A stable cache directory name.
    """
    content = json.dumps(dict(metadata), sort_keys=True).encode()

    return hashlib.sha256(content).hexdigest()
