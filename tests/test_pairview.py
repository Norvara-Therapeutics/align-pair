"""The side-by-side view, its markers and its coordinates."""

from __future__ import annotations

import pytest

from align_pair import pairview
from align_pair.aligner import Gaps, align
from align_pair.pairview import blocks, markers, render


class TestMarkers:
    def test_identical_residues_are_a_bar(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        assert markers(alignment, simple_dna) == "||||"

    def test_a_gap_is_a_space(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "AGT", simple_dna, gaps)
        assert markers(alignment, simple_dna) == "| ||"

    def test_a_positive_mismatch_is_a_colon(self, blosum62, gaps) -> None:
        alignment = align("L", "I", blosum62, gaps)
        assert markers(alignment, blosum62) == ":"

    def test_a_negative_mismatch_is_a_dot(self, blosum62, gaps) -> None:
        alignment = align("W", "D", blosum62, gaps)
        assert markers(alignment, blosum62) == "."

    def test_an_identical_pair_the_matrix_dislikes_is_still_a_bar(
        self, nuc44, gaps
    ) -> None:
        """The marker line says what the letters are, not what they score.

        S against S scores -1, and marking it as a mismatch would contradict
        the identity count beside it.
        """
        alignment = align("S", "S", nuc44, gaps)
        assert markers(alignment, nuc44) == "|"

    def test_every_marker_has_a_meaning(self) -> None:
        marks = {mark for mark, _meaning in pairview.LEGEND}
        assert marks == {"|", ":", ".", " "}

    def test_the_legend_names_each_marker(self) -> None:
        line = pairview.legend_line()
        assert "identical residues" in line
        assert "space a gap" in line


class TestBlocks:
    def test_a_short_alignment_is_one_block(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        assert len(blocks(alignment, simple_dna, 60)) == 1

    def test_a_long_alignment_is_cut_at_the_width(self, simple_dna, gaps) -> None:
        sequence = "ACGT" * 10
        alignment = align(sequence, sequence, simple_dna, gaps)
        found = blocks(alignment, simple_dna, 10)
        assert len(found) == 4
        assert all(len(block.first) == 10 for block in found)

    def test_coordinates_run_along_the_sequences(self, simple_dna, gaps) -> None:
        sequence = "ACGT" * 5
        alignment = align(sequence, sequence, simple_dna, gaps)
        found = blocks(alignment, simple_dna, 10)
        assert (found[0].start_first, found[0].end_first) == (1, 10)
        assert (found[1].start_first, found[1].end_first) == (11, 20)

    def test_gaps_do_not_advance_the_coordinate(self, simple_dna, gaps) -> None:
        alignment = align("ACCCGT", "AGT", simple_dna, gaps)
        found = blocks(alignment, simple_dna, 60)[0]
        assert (found.start_second, found.end_second) == (1, 3)
        assert (found.start_first, found.end_first) == (1, 6)

    def test_a_block_of_only_gaps_repeats_the_last_position(
        self, simple_dna, gaps
    ) -> None:
        """A row with no residues has no position of its own.

        Inventing one would put a coordinate on a residue that is not there.
        """
        alignment = align("AAAAAACGT", "ACGT", simple_dna, Gaps(0.5, 0.1))
        found = blocks(alignment, simple_dna, 5)
        assert found[0].start_second == found[0].end_second

    def test_a_local_alignment_starts_at_its_own_offset(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("TTTACGTTTT", "AAAACGTAAA", simple_dna, gaps, mode="local")
        found = blocks(alignment, simple_dna, 60)[0]
        assert found.start_first == 4
        assert found.end_first == 7

    def test_a_width_below_one_is_refused(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        with pytest.raises(ValueError, match="1 or more"):
            blocks(alignment, simple_dna, 0)


class TestReverseNumbering:
    """A reverse-complemented row is numbered along its record, descending.

    The residues shown are the record's, read the other way, so a reader
    checking the region against the file needs the numbers the file uses. The
    view and the summary have to agree about them, or one of the two is
    pointing at the wrong part of the sequence.
    """

    def test_coordinates_count_down(self, simple_dna, gaps) -> None:
        alignment = align("ACGTACGTAC", "ACGTACGTAC", simple_dna, gaps)
        found = blocks(alignment, simple_dna, 60, second_length=10)[0]
        assert (found.start_second, found.end_second) == (10, 1)
        assert (found.start_first, found.end_first) == (1, 10)

    def test_an_offset_record_is_numbered_from_its_own_length(
        self, simple_dna, gaps
    ) -> None:
        """The alignment covers positions 1-10 of a 15-base record.

        Along the reverse complement those are positions 1-10; in the record
        they are 15 down to 6.
        """
        alignment = align(
            "ACGTACGTAC", "ACGTACGTACTTTTT", simple_dna, gaps, mode="glocal"
        )
        found = blocks(alignment, simple_dna, 60, second_length=15)[0]
        assert (found.start_second, found.end_second) == (15, 6)

    def test_the_view_agrees_with_the_summary_span(self) -> None:
        """The two renderings of one interval must name the same residues."""
        from align_pair.alphabet import DNA, reverse_complement
        from align_pair.analysis import Settings, analyse
        from align_pair.reader import Record

        target = Record("t", "", reverse_complement("ACGTACGTAC", DNA) + "TTTTT")
        run = analyse(
            [Record("q", "", "ACGTACGTAC")],
            [target],
            Settings(strand="minus", mode="glocal"),
        )
        pair = run.pairs[0]
        start, end = pair.span_second
        found = blocks(
            pair.alignment, run.matrix, 60, second_length=pair.metrics.length_second
        )
        assert (found[0].start_second, found[-1].end_second) == (end, start)

    def test_each_block_continues_where_the_last_left_off(
        self, simple_dna, gaps
    ) -> None:
        sequence = "ACGT" * 5
        alignment = align(sequence, sequence, simple_dna, gaps)
        found = blocks(alignment, simple_dna, 10, second_length=20)
        assert (found[0].start_second, found[0].end_second) == (20, 11)
        assert (found[1].start_second, found[1].end_second) == (10, 1)

    def test_without_a_length_the_numbering_is_unchanged(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("ACGTACGTAC", "ACGTACGTAC", simple_dna, gaps)
        plain = blocks(alignment, simple_dna, 60)[0]
        assert (plain.start_second, plain.end_second) == (1, 10)

    def test_the_rendered_line_carries_the_descending_numbers(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("ACGTACGTAC", "ACGTACGTAC", simple_dna, gaps)
        lines = render(
            alignment, simple_dna, ("q", "t"), 60, second_length=10
        )
        assert lines[2].split() == ["t", "10", "ACGTACGTAC", "1"]


class TestRender:
    def test_each_block_is_three_lines(self, simple_dna, gaps) -> None:
        sequence = "ACGT" * 5
        alignment = align(sequence, sequence, simple_dna, gaps)
        lines = render(alignment, simple_dna, ("first", "second"), 10)
        # Twenty columns at ten to a block is two blocks: three lines each,
        # with one blank line between them.
        assert len(blocks(alignment, simple_dna, 10)) == 2
        assert len(lines) == 3 * 2 + 1

    def test_the_labels_are_written(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        lines = render(alignment, simple_dna, ("query", "target"), 60)
        assert "query" in lines[0]
        assert "target" in lines[2]

    def test_the_rows_are_aligned_in_columns(self, simple_dna, gaps) -> None:
        """The residues have to start at the same column in all three lines.

        A marker line one character out from the residues it describes is
        worse than no marker line.
        """
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        lines = render(alignment, simple_dna, ("a", "bbbbb"), 60)
        start = lines[0].index("ACGT")
        assert lines[1].index("||||") == start
        assert lines[2].index("ACGT") == start

    def test_a_long_label_is_shortened(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        lines = render(alignment, simple_dna, ("x" * 40, "y"), 60)
        assert "\u2026" in lines[0]
        assert len(lines[0]) < 60 + 40

    def test_an_empty_alignment_says_so(self, simple_dna, gaps) -> None:
        from align_pair import matrices

        matrix = matrices.simple(1.0, -5.0, "nucleotide")
        alignment = align("AAAA", "CCCC", matrix, gaps, mode="local")
        lines = render(alignment, matrix, ("a", "b"), 60)
        assert lines == ["  no residues are aligned"]


class TestFasta:
    def test_both_rows_are_written_with_their_gaps(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "AGT", simple_dna, gaps)
        text = pairview.fasta(alignment, ("a", "b"))
        assert text.splitlines() == [">a", "ACGT", ">b", "A-GT"]

    def test_the_two_records_are_the_same_length(self, simple_dna, gaps) -> None:
        alignment = align("ACCCGT", "AGT", simple_dna, gaps)
        lines = pairview.fasta(alignment, ("a", "b")).splitlines()
        assert len(lines[1]) == len(lines[3])

    def test_descriptions_are_kept(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        text = pairview.fasta(alignment, ("a", "b"), ("first one", "second one"))
        assert text.startswith(">a first one\n")

    def test_long_rows_are_wrapped(self, simple_dna, gaps) -> None:
        sequence = "ACGT" * 10
        alignment = align(sequence, sequence, simple_dna, gaps)
        lines = pairview.fasta(alignment, ("a", "b"), width=10).splitlines()
        assert lines[0] == ">a"
        assert [len(line) for line in lines[1:5]] == [10, 10, 10, 10]
        assert "".join(lines[1:5]) == sequence
        assert lines[5] == ">b"

    def test_an_empty_alignment_writes_empty_records(self, gaps) -> None:
        from align_pair import matrices

        matrix = matrices.simple(1.0, -5.0, "nucleotide")
        alignment = align("AAAA", "CCCC", matrix, gaps, mode="local")
        text = pairview.fasta(alignment, ("a", "b"))
        assert text.splitlines() == [">a", "", ">b", ""]
