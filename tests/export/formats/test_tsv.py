"""General tabular export independent of lexical resources."""

import csv
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.export import TSVWriter


@given(
    st.permutations(("name", "quantity", "description")),
    st.lists(
        st.dictionaries(
            st.sampled_from(("name", "quantity", "description")),
            st.one_of(
                st.text(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.none(),
            ),
        ),
        max_size=20,
    ),
)
def test_arbitrary_rows_preserve_values_and_column_order(
    workspace: Callable[[], Path],
    fields: list[str],
    rows: list[dict[str, str | int | float | bool | None]],
) -> None:
    """Arbitrary text, absent cells, and scalars follow the standard CSV contract."""
    path = workspace() / "inventory.tsv"

    with TSVWriter(path, fields) as writer:
        for row in rows:
            writer.write(row)

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames == fields
        assert list(reader) == [
            {
                field: "" if row.get(field) is None else str(row[field])
                for field in fields
            }
            for row in rows
        ]


def test_undeclared_columns_preserve_existing_output(
    tmp_path: Path,
) -> None:
    """A malformed row never replaces previously completed output."""
    path = tmp_path / "inventory.tsv"
    original = "name\nprevious\n"
    _ = path.write_text(original, encoding="utf-8")

    with (
        pytest.raises(ValueError, match="fields not in fieldnames"),
        TSVWriter(path, ("name",)) as writer,
    ):
        writer.write({"unexpected": "value"})

    assert path.read_text(encoding="utf-8") == original
    assert not path.with_name("inventory.tsv.part").exists()


def test_writes_require_an_open_context(
    tmp_path: Path,
) -> None:
    """Writing before entry or after closure fails without changing output."""
    path = tmp_path / "inventory.tsv"
    writer = TSVWriter(path, ("name",))

    with pytest.raises(RuntimeError, match="context manager"):
        writer.write({"name": "before"})

    with writer:
        writer.write({"name": "during"})

    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="context manager"):
        writer.write({"name": "after"})

    assert path.read_bytes() == original
