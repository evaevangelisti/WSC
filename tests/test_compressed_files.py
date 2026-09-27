"""
Exercise compressed inputs and outputs through their public readers and writers.
"""

import bz2
import gzip
import json
from collections.abc import Callable
from compression import zstd
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

    collection_manifest = cast(
        dict[str, object],
        json.loads(
            decompress((collection_dir / f"manifest.json{suffix}").read_bytes()),
        ),
    )

    collection_files = cast(dict[str, str], collection_manifest["files"])

    assert set(collection_files.values()) == {
        f"senses.jsonl{suffix}",
        f"report.json{suffix}",
        f"report.md{suffix}",
        f"manifest.json{suffix}",
    }
    for name in collection_files.values():
        assert decompress((collection_dir / name).read_bytes())

    assert tuple(read_lemmas(collection_dir / f"senses.jsonl{suffix}")) == (lemma,)

    alignment_dir = tmp_path / "alignment"
    write_alignment(
        [lemma],
        alignment_dir,
        AlignmentStatistics(AlignmentTask),
        {},
        compression,
    )

    alignment_manifest = cast(
        dict[str, object],
        json.loads(
            decompress((alignment_dir / f"manifest.json{suffix}").read_bytes()),
        ),
    )
    alignment_files = cast(dict[str, object], alignment_manifest["files"])
    report_files = cast(dict[str, str], alignment_files["reports"])

    assert alignment_files["senses"] == f"senses.jsonl{suffix}"
    assert set(report_files.values()) == {
        f"reports/{task}.json{suffix}" for task in AlignmentTask
    }
    for name in report_files.values():
        assert decompress((alignment_dir / name).read_bytes())

    assert tuple(read_lemmas(alignment_dir / f"senses.jsonl{suffix}")) == (lemma,)
