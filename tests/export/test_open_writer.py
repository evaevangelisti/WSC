"""
Tests for src/wsc/export/__init__.py.
"""

from pathlib import Path

import pytest

from wsc import export
from wsc.export import Writer, open_writer


class TestOpenWriter:
    """
    Picking a format off the suffix of the path.
    """

    # Reading the registry rather than listing it means a format added later
    # is held to this contract without a test being written for it.
    @pytest.mark.parametrize("suffix", sorted(export._WRITERS))  # pyright: ignore[reportPrivateUsage]
    def test_every_known_suffix_opens_a_writer(
        self,
        tmp_path: Path,
        suffix: str,
    ) -> None:
        """Every format the registry offers opens a writer."""
        assert isinstance(open_writer(tmp_path / f"senses{suffix}"), Writer)

    def test_reads_the_suffix_whatever_its_case(
        self,
        tmp_path: Path,
    ) -> None:
        """A suffix in capitals names the same format as one in lowercase."""
        assert isinstance(open_writer(tmp_path / "SENSES.JSONL"), Writer)

    def test_opens_nothing_yet(
        self,
        tmp_path: Path,
    ) -> None:
        """The file is opened when the writer is entered, not when it is built."""
        output_path = tmp_path / "senses.jsonl"

        _ = open_writer(output_path)

        assert not output_path.exists()

    def test_refuses_a_suffix_it_does_not_know(
        self,
        tmp_path: Path,
    ) -> None:
        """The refusal names the formats there are, since one of them is the answer."""
        with pytest.raises(ValueError, match=r"Unknown format '\.parquet'.+\.jsonl"):
            _ = open_writer(tmp_path / "senses.parquet")

    def test_refuses_a_path_with_no_suffix_at_all(
        self,
        tmp_path: Path,
    ) -> None:
        """A name with no suffix names no format."""
        with pytest.raises(ValueError, match="Unknown format"):
            _ = open_writer(tmp_path / "senses")
