"""
Tests for src/wsc/models/pos.py.
"""

import pytest

from wsc.models import POS


class TestPOS:
    """
    The parts of speech the collector keeps.
    """

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("noun", POS.NOUN),
            ("verb", POS.VERB),
            ("adj", POS.ADJECTIVE),
            ("adv", POS.ADVERB),
        ],
    )
    def test_converts_from_a_wiktextract_code(
        self,
        code: str,
        expected: POS,
    ) -> None:
        """The values are wiktextract's own, so its codes convert directly."""
        assert POS(code) is expected

    def test_refuses_a_part_of_speech_it_does_not_keep(
        self,
    ) -> None:
        """The refusal is what the extractor reads to skip an entry."""
        with pytest.raises(ValueError, match="not a valid POS"):
            _ = POS("intj")
