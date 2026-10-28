"""The dynamic programme, checked against alignments worked out by hand."""

from __future__ import annotations

import math

import pytest

from align_pair import matrices
from align_pair.aligner import (
    CO_OPTIMAL_CAP,
    MODE_SUMMARY,
    MODES,
    AlignmentError,
    Gaps,
    align,
    band_diagonals,
    cells,
)


class TestGaps:
    def test_a_gap_of_one_residue_costs_the_opening_penalty(self) -> None:
        assert Gaps(10.0, 0.5).cost(1) == 10.0

    def test_each_further_residue_costs_the_extension(self) -> None:
        gaps = Gaps(10.0, 0.5)
        assert gaps.cost(2) == 10.5
        assert gaps.cost(5) == 12.0

    def test_a_gap_of_no_residues_costs_nothing(self) -> None:
        assert Gaps(10.0, 0.5).cost(0) == 0.0
        assert Gaps(10.0, 0.5).cost(-3) == 0.0

    def test_equal_penalties_are_a_linear_gap(self) -> None:
        gaps = Gaps(2.0, 2.0)
        assert [gaps.cost(length) for length in (1, 2, 3)] == [2.0, 4.0, 6.0]

    def test_a_negative_penalty_is_refused(self) -> None:
        with pytest.raises(AlignmentError, match="must not be negative"):
            Gaps(-1.0, 0.5)

    def test_a_penalty_that_is_not_a_number_is_refused(self) -> None:
        with pytest.raises(AlignmentError, match="finite"):
            Gaps(math.nan, 0.5)
        with pytest.raises(AlignmentError, match="finite"):
            Gaps(10.0, math.inf)

    def test_free_says_whether_gaps_cost_anything(self) -> None:
        assert Gaps(0.0, 0.0).free
        assert not Gaps(0.0, 0.5).free
        assert not Gaps(10.0, 0.5).free


class TestGlobal:
    def test_two_identical_sequences_align_without_gaps(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "ACGT", simple_dna, gaps)
        assert alignment.aligned_first == "ACGT"
        assert alignment.aligned_second == "ACGT"
        assert alignment.score == 4.0
        assert alignment.co_optimal == 1

    def test_a_single_deletion_is_placed_as_one_gap(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "AGT", simple_dna, gaps)
        assert alignment.aligned_first == "ACGT"
        assert alignment.aligned_second == "A-GT"
        assert alignment.score == 3.0 - 2.0

    def test_a_longer_gap_costs_opening_once(self, simple_dna) -> None:
        alignment = align("ACCCGT", "AGT", simple_dna, Gaps(2.0, 1.0))
        assert alignment.aligned_second == "A---GT"
        assert alignment.score == 3.0 - (2.0 + 2 * 1.0)

    def test_every_residue_of_both_sequences_is_in_the_alignment(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("AAAA", "CC", simple_dna, gaps)
        assert alignment.aligned_first.replace("-", "") == "AAAA"
        assert alignment.aligned_second.replace("-", "") == "CC"
        assert alignment.start_first == 1
        assert alignment.end_first == 4

    def test_terminal_gaps_are_paid_for(self, simple_dna, gaps) -> None:
        alignment = align("AAACGT", "CGT", simple_dna, gaps)
        assert alignment.score == 3.0 - gaps.cost(3)


class TestGlocal:
    def test_the_first_sequence_is_aligned_end_to_end(self, simple_dna, gaps) -> None:
        alignment = align("CGT", "AAACGTAAA", simple_dna, gaps, mode="glocal")
        assert alignment.aligned_first == "CGT"
        assert alignment.aligned_second == "CGT"
        assert alignment.start_first == 1
        assert alignment.end_first == 3

    def test_the_overhangs_of_the_second_are_free(self, simple_dna, gaps) -> None:
        alignment = align("CGT", "AAACGTAAA", simple_dna, gaps, mode="glocal")
        assert alignment.score == 3.0
        assert (alignment.start_second, alignment.end_second) == (4, 6)

    def test_gaps_in_the_first_sequence_are_still_paid(self, simple_dna, gaps) -> None:
        alignment = align("CGT", "CGAT", simple_dna, gaps, mode="glocal")
        assert alignment.score == 3.0 - gaps.open

    def test_an_empty_second_sequence_still_accounts_for_the_first(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("CGT", "", simple_dna, gaps, mode="glocal")
        assert alignment.score == -gaps.cost(3)


class TestOverlap:
    def test_a_suffix_aligns_to_a_prefix_for_free(self, simple_dna, gaps) -> None:
        alignment = align("TTTTACGT", "ACGTAAAA", simple_dna, gaps, mode="overlap")
        assert alignment.aligned_first == "ACGT"
        assert alignment.aligned_second == "ACGT"
        assert alignment.score == 4.0
        assert (alignment.start_first, alignment.end_first) == (5, 8)
        assert (alignment.start_second, alignment.end_second) == (1, 4)

    def test_overhangs_at_both_ends_are_free(self, simple_dna, gaps) -> None:
        alignment = align("ACGT", "TTACGTTT", simple_dna, gaps, mode="overlap")
        assert alignment.score == 4.0
        assert alignment.aligned_first == "ACGT"


class TestLocal:
    def test_only_the_best_scoring_block_is_reported(self, simple_dna, gaps) -> None:
        alignment = align("TTTACGTTTT", "AAAACGTAAA", simple_dna, gaps, mode="local")
        assert alignment.aligned_first == "ACGT"
        assert alignment.aligned_second == "ACGT"
        assert alignment.score == 4.0

    def test_an_alignment_begins_and_ends_with_an_aligned_pair(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("ACGTACGT", "ACGTTTACGT", simple_dna, gaps, mode="local")
        assert alignment.aligned_first[0] != "-"
        assert alignment.aligned_second[0] != "-"
        assert alignment.aligned_first[-1] != "-"
        assert alignment.aligned_second[-1] != "-"

    def test_nothing_worth_aligning_gives_the_empty_alignment(self, gaps) -> None:
        matrix = matrices.simple(1.0, -5.0, "nucleotide")
        alignment = align("AAAA", "CCCC", matrix, gaps, mode="local")
        assert alignment.empty
        assert alignment.score == 0.0
        assert alignment.co_optimal == 1
        assert (alignment.start_first, alignment.end_first) == (0, 0)

    def test_a_zero_scoring_optimum_is_the_empty_alignment(self, gaps) -> None:
        """The empty alignment is the optimum unless something scores ABOVE zero.

        A pair of subsequences scoring exactly zero is that same alignment
        written out, not a second one tying with it, so reporting it as a
        non-empty alignment would contradict the policy and count an alignment
        the co-optimal count deliberately excludes.
        """
        matrix = matrices.simple(0.0, -1.0, "nucleotide")
        alignment = align("A", "A", matrix, gaps, mode="local")
        assert alignment.empty
        assert alignment.score == 0.0
        assert alignment.co_optimal == 1

    def test_something_above_zero_is_reported(self, simple_dna, gaps) -> None:
        alignment = align("A", "A", simple_dna, gaps, mode="local")
        assert not alignment.empty
        assert alignment.score == 1.0

    def test_coordinates_point_into_the_original_sequences(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("TTTACGTTTT", "AAAACGTAAA", simple_dna, gaps, mode="local")
        assert (alignment.start_first, alignment.end_first) == (4, 7)
        assert (alignment.start_second, alignment.end_second) == (4, 7)


class TestCoOptimal:
    def test_a_unique_alignment_is_reported_as_one(self, simple_dna, gaps) -> None:
        assert align("ACGT", "ACGT", simple_dna, gaps).co_optimal == 1

    def test_a_gap_that_could_sit_either_side_counts_twice(
        self, simple_dna, gaps
    ) -> None:
        # Deleting either A of 'AA' gives the same alignment score and a
        # different alignment.
        alignment = align("AA", "A", simple_dna, gaps)
        assert alignment.co_optimal == 2

    def test_the_count_is_not_capped_for_small_inputs(self, simple_dna, gaps) -> None:
        alignment = align("ACGTACGT", "ACGTACGT", simple_dna, gaps)
        assert not alignment.co_optimal_capped
        assert alignment.co_optimal == 1

    def test_free_gaps_tie_every_placement_of_a_necessary_gap(
        self, simple_dna
    ) -> None:
        """Free gaps do not make gaps attractive; they make placements tie.

        A gap still loses whatever match it displaces, so the gapless
        alignment of two identical sequences remains the unique optimum. What
        free gaps do is stop the arithmetic distinguishing one position for an
        unavoidable gap from another: 'AAAA' against 'AA' has to leave two
        residues unpaired, and every way of doing it scores the same.
        """
        assert align("A" * 8, "A" * 8, simple_dna, Gaps(0.0, 0.0)).co_optimal == 1
        assert align("AAA", "AA", simple_dna, Gaps(0.0, 0.0)).co_optimal == 3
        assert align("AAAA", "AA", simple_dna, Gaps(0.0, 0.0)).co_optimal == 6

    def test_the_count_is_a_bound_exactly_when_it_reached_the_cap(
        self, simple_dna
    ) -> None:
        """The bound travels with the count, not with the run.

        A count that reached the cap is held at the cap, so any sum drawing on
        it is capped in turn. A run in which some state unrelated to the
        optimum saturated says nothing about the optimum: fifty identical
        residues under free gaps have exactly one optimal alignment, and
        reporting 'at least 1' would be false.
        """
        for first, second, gaps in (
            ("A" * 50, "A" * 50, Gaps(0.0, 0.0)),
            ("A" * 60, "A" * 30, Gaps(0.0, 0.0)),
            ("ACGT", "ACGT", Gaps(2.0, 1.0)),
            ("AAAA", "AA", Gaps(0.0, 0.0)),
        ):
            alignment = align(first, second, simple_dna, gaps)
            assert alignment.co_optimal <= CO_OPTIMAL_CAP
            assert alignment.co_optimal_capped == (
                alignment.co_optimal == CO_OPTIMAL_CAP
            )

    def test_a_unique_optimum_is_never_reported_as_a_bound(
        self, simple_dna
    ) -> None:
        alignment = align("A" * 50, "A" * 50, simple_dna, Gaps(0.0, 0.0))
        assert alignment.co_optimal == 1
        assert not alignment.co_optimal_capped


class TestBand:
    def test_a_band_wide_enough_gives_the_same_answer(self, simple_dna, gaps) -> None:
        first, second = "ACGTACGTAC", "ACGTTCGTAC"
        full = align(first, second, simple_dna, gaps)
        banded = align(first, second, simple_dna, gaps, band=len(first))
        assert banded.score == full.score
        assert banded.aligned_first == full.aligned_first
        assert not banded.band_limited

    def test_a_band_never_beats_the_full_matrix(self, simple_dna, gaps) -> None:
        first, second = "ACGTTTTTTTACGT", "ACGTACGT"
        full = align(first, second, simple_dna, gaps)
        for width in range(0, 6):
            banded = align(first, second, simple_dna, gaps, band=width)
            assert banded.score <= full.score

    def test_a_partial_band_says_its_score_is_a_lower_bound(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("ACGTACGTAC", "ACGTACGTAC", simple_dna, gaps, band=2)
        assert alignment.band == 2
        assert alignment.band_limited
        assert alignment.band_margin is not None

    def test_the_band_holds_a_complete_path_however_narrow(self, simple_dna) -> None:
        alignment = align("AAAAAAAA", "AA", simple_dna, Gaps(2.0, 1.0), band=0)
        assert alignment.aligned_first.replace("-", "") == "AAAAAAAA"
        assert alignment.aligned_second.replace("-", "") == "AA"

    def test_the_band_spans_the_length_difference(self) -> None:
        assert band_diagonals(10, 10, 2) == (-2, 2)
        assert band_diagonals(10, 14, 2) == (-2, 6)
        assert band_diagonals(14, 10, 2) == (-6, 2)

    def test_a_band_is_refused_outside_global_mode(self, simple_dna, gaps) -> None:
        for mode in ("glocal", "overlap", "local"):
            with pytest.raises(AlignmentError, match="only defined for global"):
                align("ACGT", "ACGT", simple_dna, gaps, mode=mode, band=2)

    def test_a_negative_band_is_refused(self, simple_dna, gaps) -> None:
        with pytest.raises(AlignmentError, match="must not be negative"):
            align("ACGT", "ACGT", simple_dna, gaps, band=-1)


class TestCells:
    def test_an_unbanded_run_fills_the_whole_matrix(self) -> None:
        assert cells(10, 20) == 11 * 21

    def test_a_band_fills_fewer_cells(self) -> None:
        assert cells(100, 100, band=5) < cells(100, 100)

    def test_the_count_matches_what_the_run_reports(self, simple_dna, gaps) -> None:
        alignment = align("ACGTACG", "ACGTAC", simple_dna, gaps)
        assert alignment.cells == cells(7, 6)
        banded = align("ACGTACG", "ACGTAC", simple_dna, gaps, band=2)
        assert banded.cells == cells(7, 6, band=2)


class TestEdges:
    def test_two_empty_sequences_give_an_empty_alignment(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("", "", simple_dna, gaps)
        assert alignment.empty
        assert alignment.score == 0.0
        assert alignment.co_optimal == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_an_empty_sequence_gives_one_alignment(
        self, mode: str, simple_dna, gaps
    ) -> None:
        alignment = align("ACGT", "", simple_dna, gaps, mode=mode)
        assert alignment.co_optimal == 1

    def test_an_unknown_mode_is_refused(self, simple_dna, gaps) -> None:
        with pytest.raises(AlignmentError, match="unknown alignment mode"):
            align("ACGT", "ACGT", simple_dna, gaps, mode="semiglobal")

    def test_every_mode_has_a_summary(self) -> None:
        assert set(MODE_SUMMARY) == set(MODES)
        for summary in MODE_SUMMARY.values():
            assert summary and not summary.endswith(".")

    def test_a_residue_the_matrix_cannot_score_is_named(self, gaps) -> None:
        matrix = matrices.load("BLOSUM62")
        with pytest.raises(matrices.MatrixError, match="'U'"):
            align("ACDU", "ACDE", matrix, gaps)

    def test_the_length_of_an_alignment_is_its_column_count(
        self, simple_dna, gaps
    ) -> None:
        alignment = align("ACGT", "AGT", simple_dna, gaps)
        assert alignment.length == 4
        assert len(alignment) == 4
        assert len(alignment.columns) == 4
