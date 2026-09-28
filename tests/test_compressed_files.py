"""
Exercise compressed inputs and outputs through their public readers and writers.
"""

import bz2
import gzip
import json
import tarfile
from collections.abc import Callable
from compression import zstd
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest

from wsc.alignment.reporting import AlignmentStatistics, write_alignment
from wsc.collection import write_collection
from wsc.constants import ALIGNMENT_FIELDS, PROMPTS_PATH
from wsc.export import TSVWriter, Writer, open_writer
from wsc.extract.dump.wikidata import read_wikidata_ids, write_wikidata_ids
from wsc.extract.wiktionary.off_page_translations import (
    read_off_page_translations,
    write_off_page_translations,
)
from wsc.files import Compression
from wsc.models import POS, Lemma
from wsc.models.alignment import AlignmentTask
from wsc.reading import read_alignment_cache, read_lemmas, read_prompts, read_synsets


@pytest.mark.parametrize(
    ("suffix", "compress", "decompress", "signature", "compression"),
    [
        (".gz", gzip.compress, gzip.decompress, b"\x1f\x8b", Compression.GZIP),
        (".bz2", bz2.compress, bz2.decompress, b"BZh", Compression.BZIP2),
        (
            ".zst",
            zstd.compress,
            zstd.decompress,
            b"\x28\xb5\x2f\xfd",
            Compression.ZSTANDARD,
        ),
    ],
)
def test_compressed_pipeline_files(
    tmp_path: Path,
    suffix: str,
    compress: Callable[[bytes], bytes],
    decompress: Callable[[bytes], bytes],
    signature: bytes,
    compression: Compression,
) -> None:
    """
    Compressed inputs and staged outputs retain their formats and contents.
    """
    synsets_path = tmp_path / f"synsets.jsonl{suffix}"
    _ = synsets_path.write_bytes(
        compress(b'{"pos":"noun","members":["word"],"glosses":["A word."]}\n'),
    )

    assert next(read_synsets(synsets_path)).members[0].lemma == "word"

    prompts_path = tmp_path / f"prompts.toml{suffix}"
    _ = prompts_path.write_bytes(compress(PROMPTS_PATH.read_bytes()))

    assert read_prompts(prompts_path).tasks == read_prompts(PROMPTS_PATH).tasks
    assert read_prompts(prompts_path).name == PROMPTS_PATH.stem

    lemma = Lemma("word.noun", "word", POS.NOUN)
    collection_path = tmp_path / f"senses.jsonl{suffix}"
    collection_writer: Writer[Lemma] = open_writer(collection_path)

    with collection_writer as writer:
        writer.write(lemma)

    assert collection_path.read_bytes().startswith(signature)
    assert tuple(read_lemmas(collection_path)) == (lemma,)
    assert not collection_path.with_name(f"{collection_path.name}.part").exists()

    table_path = tmp_path / f"synsets.tsv{suffix}"

    with TSVWriter(table_path, ALIGNMENT_FIELDS) as writer:
        writer.write({"alignment_id": "query", "source_id": "sense"})

    assert table_path.read_bytes().startswith(signature)
    assert set(read_alignment_cache(table_path)["query"]) == {"sense"}

    identifiers_path = tmp_path / f"wikidata-ids.json{suffix}"
    write_wikidata_ids(identifiers_path, {"sense": ("Q42",)})

    assert identifiers_path.read_bytes().startswith(signature)
    assert read_wikidata_ids(identifiers_path) == {"sense": ("Q42",)}

    translations_path = tmp_path / f"off-page-translations.json{suffix}"
    write_off_page_translations(translations_path, {})

    assert translations_path.read_bytes().startswith(signature)
    assert read_off_page_translations(translations_path) == {}

    collection_dir = tmp_path / "collection"
    write_collection([lemma], collection_dir, {}, compression)
    collection_archive_path = tmp_path / f"collection.tar{suffix}"

    assert not collection_dir.exists()
    assert collection_archive_path.read_bytes().startswith(signature)

    with tarfile.open(
        fileobj=BytesIO(decompress(collection_archive_path.read_bytes())), mode="r:"
    ) as archive:
        collection_files = {
            member.name: content.read()
            for member in archive
            if member.isfile()
            if (content := archive.extractfile(member)) is not None
        }

    collection_manifest = cast(
        dict[str, object], json.loads(collection_files["collection/manifest.json"])
    )

    manifest_files = cast(dict[str, str], collection_manifest["files"])

    assert set(manifest_files.values()) == {
        "senses.jsonl",
        "report.json",
        "report.md",
        "manifest.json",
    }
    for name in manifest_files.values():
        assert collection_files[f"collection/{name}"]

    assert tuple(read_lemmas(collection_archive_path)) == (lemma,)

    alignment_dir = tmp_path / "alignment"
    write_alignment(
        [lemma],
        alignment_dir,
        AlignmentStatistics(AlignmentTask),
        {},
        compression,
    )
    alignment_archive_path = tmp_path / f"alignment.tar{suffix}"

    assert not alignment_dir.exists()
    assert alignment_archive_path.read_bytes().startswith(signature)

    with tarfile.open(
        fileobj=BytesIO(decompress(alignment_archive_path.read_bytes())), mode="r:"
    ) as archive:
        alignment_files = {
            member.name: content.read()
            for member in archive
            if member.isfile()
            if (content := archive.extractfile(member)) is not None
        }

    alignment_manifest = cast(
        dict[str, object], json.loads(alignment_files["alignment/manifest.json"])
    )
    manifest_files = cast(dict[str, object], alignment_manifest["files"])
    report_files = cast(dict[str, str], manifest_files["reports"])

    assert manifest_files["senses"] == "senses.jsonl"
    assert set(report_files.values()) == {
        f"reports/{task}.json" for task in AlignmentTask
    }
    for name in report_files.values():
        assert alignment_files[f"alignment/{name}"]

    assert tuple(read_lemmas(alignment_archive_path)) == (lemma,)
