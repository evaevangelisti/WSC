"""Output paths shared by collection writing and its command line."""

from pathlib import Path

COLLECTION_DIR = Path("collection")
"""Default destination for collected entries and their reports."""

COLLECTION_FILES = {
    "senses": "senses.jsonl",
    "statistics": "report.json",
    "report": "report.md",
    "manifest": "manifest.json",
}
"""Collection filenames ordered for publication, with the manifest last."""
