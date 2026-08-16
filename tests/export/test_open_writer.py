"""
Tests for src/wsc/export/__init__.py.

Only the choosing is tested here. What each format writes is tested beside
that format, which is where a format added later is tested too.
"""

import string
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.export import Writer, open_writer

# The formats the collector offers. A format added later is added here, and
# brings a file of its own under formats/.
_KNOWN = [".jsonl"]

_CASES = st.lists(st.booleans(), min_size=5, max_size=5).map(
    lambda capitals: ".{}".format(
        "".join(
            letter.upper() if capital else letter
            for letter, capital in zip("jsonl", capitals, strict=True)
        )
    )
)

_UNKNOWN = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6).filter(
    lambda suffix: f".{suffix}" not in _KNOWN
)


class TestOpenWriter:
    """
    Picking a format off the suffix of the path.
    """

    @pytest.mark.parametrize("suffix", _KNOWN)
    def test_every_format_the_collector_offers_opens_a_writer(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """A format is offered by being written, so each one hands back a writer."""
        assert isinstance(open_writer(workspace() / f"senses{suffix}"), Writer)

    @given(_CASES)
    def test_reads_the_suffix_whatever_its_case(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """A suffix in capitals names the same format as one in lowercase."""
        assert isinstance(open_writer(workspace() / f"senses{suffix}"), Writer)

    @pytest.mark.parametrize("suffix", _KNOWN)
    def test_opens_nothing_yet(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """The file is opened when the writer is entered, not when it is built."""
        directory = workspace()

        _ = open_writer(directory / f"senses{suffix}")

        assert list(directory.iterdir()) == []

    @given(_UNKNOWN)
    def test_refuses_a_suffix_it_does_not_know(
        self,
        workspace: Callable[[], Path],
        suffix: str,
    ) -> None:
        """The refusal names the formats there are, since one of them is the answer."""
        with pytest.raises(ValueError, match="Unknown format") as refusal:
            _ = open_writer(workspace() / f"senses.{suffix}")

        assert suffix in str(refusal.value)
        assert all(known in str(refusal.value) for known in _KNOWN)

    def test_refuses_a_path_with_no_suffix_at_all(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """A name with no suffix names no format."""
        with pytest.raises(ValueError, match="Unknown format"):
            _ = open_writer(workspace() / "senses")
