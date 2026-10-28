"""The tabular outputs: one table, rendered the same way wherever it appears."""

from __future__ import annotations

import math

from align_pair import matrices, rows
from align_pair.analysis import Settings, analyse
from align_pair.reader import Record
from align_pair.rows import (
    COLUMN_COLUMNS,
    MATRIX_COLUMNS,
    PAIR_COLUMNS,
    all_column_rows,
    column_rows,
    field,
    matrix_rows,
    pair_rows,
    render,
)


def run(first: str, second: str, **settings):
    return analyse(
        [Record("one", "", first)], [Record("two", "", second)], Settings(**settings)
    )


class TestField:
    def test_none_is_an_empty_field(self) -> None:
        """An absent value is not a zero.

        An identity of 0% and no identity at all are different facts, and a
        column of zeros would hide which was meant.
        """
        assert field(None) == ""

    def test_a_whole_float_loses_its_decimal_point(self) -> None:
        assert field(429.0) == "429"

    def test_a_fractional_float_keeps_its_precision(self) -> None:
        assert field(1 / 3) == repr(1 / 3)

    def test_a_boolean_is_written_as_a_word(self) -> None:
        assert field(True) == "true"
        assert field(False) == "false"

    def test_a_tab_inside_a_description_becomes_a_space(self) -> None:
        """One row has to stay one row.

        A FASTA description may hold a tab, and writing it through would add a
        column to every reader of the file.
        """
        assert field("some\tthing") == "some thing"
        assert field("two\nlines") == "two lines"

    def test_an_integer_is_written_plainly(self) -> None:
        assert field(42) == "42"


class TestRender:
    def test_the_header_comes_first(self) -> None:
        text = render(("a", "b"), [[1, 2]])
        assert text.splitlines()[0] == "a\tb"

    def test_a_table_ends_with_a_newline(self) -> None:
        assert render(("a",), [[1]]).endswith("\n")

    def test_an_empty_table_is_just_its_header(self) -> None:
        assert render(("a", "b"), []) == "a\tb\n"


class TestPairRows:
    def test_a_row_is_written_for_every_pair(self) -> None:
        analysis = analyse(
            [Record("a", "", "ACGTACGTACGTACGTACGTACGT")],
            [
                Record("b", "", "ACGTACGTACGTACGTACGTACGT"),
                Record("c", "", "ACGTACGTACGTACGTACGTACGA"),
            ],
            Settings(),
        )
        assert len(pair_rows(analysis)) == 2

    def test_every_column_is_filled(self) -> None:
        analysis = run("ACGTACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTACGA")
        for row in pair_rows(analysis):
            assert len(row) == len(PAIR_COLUMNS)

    def test_the_identity_columns_name_their_denominator(self) -> None:
        """A bare 'identity' column would be the thing this package avoids."""
        assert "identity_of_alignment" in PAIR_COLUMNS
        assert "identity_of_shorter" in PAIR_COLUMNS
        assert "identity" not in PAIR_COLUMNS

    def test_the_values_are_unrounded(self) -> None:
        analysis = run("ACGTACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTACGA")
        text = render(PAIR_COLUMNS, pair_rows(analysis))
        header, row = text.splitlines()
        values = dict(zip(header.split("\t"), row.split("\t")))
        identity = float(values["identity_of_alignment"])
        assert identity == 23 / 24
        assert values["identity_of_alignment"] != "0.96"

    def test_absent_coordinates_are_empty_not_zero(self) -> None:
        matrix = matrices.simple(1.0, -5.0, "nucleotide")
        del matrix
        analysis = run("AAAAAAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCCCCCC",
                       mode="local", matrix="simple", mismatch=-5.0)
        row = pair_rows(analysis)[0]
        fields = dict(zip(PAIR_COLUMNS, row))
        assert fields["start_first"] is None
        assert field(fields["start_first"]) == ""

    def test_the_band_columns_are_empty_when_unbanded(self) -> None:
        analysis = run("ACGTACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTACGA")
        fields = dict(zip(PAIR_COLUMNS, pair_rows(analysis)[0]))
        assert fields["band"] is None
        assert fields["band_margin"] is None


class TestColumnRows:
    def test_one_row_per_column(self) -> None:
        analysis = run("ACGT", "AGT", matrix="simple")
        found = column_rows(analysis.pairs[0], analysis.matrix)
        assert len(found) == analysis.pairs[0].metrics.length

    def test_every_column_is_filled(self) -> None:
        analysis = run("ACGT", "AGT", matrix="simple")
        for row in column_rows(analysis.pairs[0], analysis.matrix):
            assert len(row) == len(COLUMN_COLUMNS)

    def test_a_gap_column_has_no_score(self) -> None:
        analysis = run("ACGT", "AGT", matrix="simple")
        found = column_rows(analysis.pairs[0], analysis.matrix)
        fields = [dict(zip(COLUMN_COLUMNS, row)) for row in found]
        gapped = [row for row in fields if "-" in (row["residue_first"], row["residue_second"])]
        assert gapped
        for row in gapped:
            assert row["score"] is None

    def test_positions_advance_only_where_a_residue_sits(self) -> None:
        analysis = run("ACGT", "AGT", matrix="simple")
        found = [dict(zip(COLUMN_COLUMNS, row)) for row in column_rows(
            analysis.pairs[0], analysis.matrix
        )]
        first = [row["position_first"] for row in found]
        second = [row["position_second"] for row in found]
        assert first == [1, 2, 3, 4]
        assert second == [1, None, 2, 3]

    def test_the_scores_add_up_to_the_alignment_score(self) -> None:
        """The column table is a decomposition, not a parallel calculation."""
        analysis = run("ACGTACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGTACGA")
        pair = analysis.pairs[0]
        found = [dict(zip(COLUMN_COLUMNS, row)) for row in column_rows(
            pair, analysis.matrix
        )]
        total = sum(row["score"] for row in found if row["score"] is not None)
        gaps = analysis.settings.gaps
        charged = 0.0
        previous = False
        for row in found:
            if row["score"] is None:
                charged += gaps.extend if previous else gaps.open
                previous = True
            else:
                previous = False
        assert math.isclose(total - charged, pair.alignment.score)

    def test_every_pair_is_included(self) -> None:
        analysis = analyse(
            [Record("a", "", "ACGTACGTACGTACGTACGTACGT")],
            [
                Record("b", "", "ACGTACGTACGTACGTACGTACGT"),
                Record("c", "", "ACGTACGTACGTACGTACGTACGA"),
            ],
            Settings(),
        )
        identifiers = {row[1] for row in all_column_rows(analysis)}
        assert identifiers == {"b", "c"}


class TestMatrixRows:
    def test_every_built_in_is_listed(self) -> None:
        found = matrix_rows(matrices.listing())
        assert len(found) == len(matrices.BUILT_IN)
        for row in found:
            assert len(row) == len(MATRIX_COLUMNS)

    def test_the_range_is_written_plainly(self) -> None:
        found = {row[0]: row for row in matrix_rows(matrices.listing())}
        assert found["BLOSUM62"][3] == "-4 to 11"


class TestSeparator:
    def test_the_separator_is_a_tab(self) -> None:
        assert rows.SEPARATOR == "\t"
