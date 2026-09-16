"""
Tests for src/wsc/export/__init__.py.

Only the choosing is tested here. What each format writes is tested beside that format,
which is where a format added later is tested too.
"""

import string
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.export import Writer, open_writer
from wsc.models import Lemma

_KNOWN = [".jsonl"]

_CASES = st.lists(st.booleans(), min_size=5, max_size=5).map(
    lambda capitals: ".{}".format(
        "".join(
            letter.upper() if capital else letter
            for letter, capital in zip("jsonl", capitals, strict=True)
        ),
    ),
)

_UNKNOWN = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6).filter(
    lambda suffix: f".{suffix}" not in _KNOWN,
)


class TestOpenWriter:
    """Picking a format off the suffix of the path."""

    @given(_CASES)
    def test_accepts_mixed_case(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """A suffix in capitals names the same format as one in lowercase."""
        writer: Writer[Lemma] = open_writer(workspace() / f"senses{suffix}")

        assert isinstance(writer, Writer)

    @pytest.mark.parametrize("suffix", _KNOWN)
    def test_defers_file_creation(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """Entering the writer opens the file."""
        directory = workspace()

        _: Writer[Lemma] = open_writer(directory / f"senses{suffix}")

        assert list(directory.iterdir()) == []

    @given(_UNKNOWN)
    def test_rejects_unknown_suffix(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """The refusal names the formats there are, since one of them is the answer."""
        with pytest.raises(ValueError, match="Unknown format") as refusal:
            _: Writer[Lemma] = open_writer(workspace() / f"senses.{suffix}")

        assert suffix in str(refusal.value)
        assert all(known in str(refusal.value) for known in _KNOWN)

    def test_rejects_missing_suffix(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """A name with no suffix names no format."""
        with pytest.raises(ValueError, match="Unknown format"):
            _: Writer[Lemma] = open_writer(workspace() / "senses")
