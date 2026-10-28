"""What an alignment contains, counted column by column."""

from __future__ import annotations

import pytest

from align_pair import matrices
from align_pair.aligner import Gaps, align
from align_pair.alphabet import DNA, PROTEIN
from align_pair.metrics import Metrics, measure


def _measure(first: str, second: str, matrix, alphabet, mode: str = "global"):
    alignment = align(first, second, matrix, Gaps(2.0, 1.0), mode=mode)
    return alignment, measure(
        alignment, matrix, alphabet, len(first), len(second)
    )


class TestCounts:
    def test_identical_sequences_are_wholly_identical(self, simple_dna) -> None:
        _alignment, metrics = _measure("ACGT", "ACGT", simple_dna, DNA)
        assert metrics.length == 4
        assert metrics.identities == 4
        assert metrics.mismatches == 0
        assert metrics.gaps == 0
        assert metrics.identity == 1.0

    def test_a_mismatch_is_counted_as_one(self, simple_dna) -> None:
        _alignment, metrics = _measure("ACGT", "ACCT", simple_dna, DNA)
        assert metrics.identities == 3
        assert metrics.mismatches == 1
        assert metrics.identity == pytest.approx(0.75)

    def test_gap_columns_are_counted_by_which_row_they_fall_in(
        self, simple_dna
    ) -> None:
        """A gap is counted against the row the gap character sits in.

        Aligning 'ACGT' with 'AGT' writes 'A-GT' in the second row, so the gap
        is a gap in the second sequence, not in the first.
        """
        alignment, metrics = _measure("ACGT", "AGT", simple_dna, DNA)
        assert alignment.aligned_second == "A-GT"
        assert metrics.gaps == 1
        assert metrics.gaps_first == 0
        assert metrics.gaps_second == 1

    def test_the_other_row_is_counted_the_other_way(self, simple_dna) -> None:
        alignment, metrics = _measure("AGT", "ACGT", simple_dna, DNA)
        assert alignment.aligned_first == "A-GT"
        assert metrics.gaps_first == 1
        assert metrics.gaps_second == 0

    def test_gap_runs_are_counted_separately_from_columns(self, simple_dna) -> None:
        alignment, metrics = _measure("ACCCGT", "AGT", simple_dna, DNA)
        assert metrics.gaps == 3
        assert metrics.gap_runs == 1
        assert metrics.longest_gap == 3
        assert alignment.aligned_second == "A---GT"

    def test_two_runs_are_two_runs(self, simple_dna) -> None:
        metrics = measure(
            _fake("AC-GT-A", "ACTGTCA"), simple_dna, DNA, 5, 7
        )
        assert metrics.gaps == 2
        assert metrics.gap_runs == 2
        assert metrics.longest_gap == 1

    def test_aligned_columns_exclude_gaps(self, simple_dna) -> None:
        _alignment, metrics = _measure("ACGT", "AGT", simple_dna, DNA)
        assert metrics.aligned_columns == 3

    def test_an_empty_alignment_counts_nothing(self, simple_dna) -> None:
        matrix = matrices.simple(1.0, -5.0, "nucleotide")
        _alignment, metrics = _measure("AAAA", "CCCC", matrix, DNA, mode="local")
        assert metrics.length == 0
        assert metrics.identities == 0


class TestGapAccounting:
    """The gap counts have to agree with the score the alignment reports.

    A gap run is what the opening penalty is charged for, so the number of runs
    is not a presentational detail: if it disagrees with the score, one of the
    two is wrong and the reader cannot tell which.
    """

    def test_opposite_direction_gaps_are_two_runs(self) -> None:
        """A gap in one sequence then a gap in the other is two runs.

        They are adjacent columns but not one gap: each opens, so each pays the
        opening penalty, and 'A' against 'C' under a punishing mismatch is
        resolved as two single-residue gaps rather than one of two.
        """
        matrix = matrices.simple(1.0, -100.0, "nucleotide")
        alignment, metrics = _measure("A", "C", matrix, DNA)
        assert (alignment.aligned_first, alignment.aligned_second) == ("-A", "C-")
        assert metrics.gaps == 2
        assert metrics.gap_runs == 2
        assert metrics.longest_gap == 1
        assert metrics.gaps_first == 1
        assert metrics.gaps_second == 1

    def test_same_direction_gaps_are_one_run(self) -> None:
        metrics = measure(_fake("ACCCGT", "A---GT"), _simple(), DNA, 6, 3)
        assert metrics.gap_runs == 1
        assert metrics.longest_gap == 3

    @pytest.mark.parametrize(
        "first,second",
        [
            ("ACGT", "ACGT"),
            ("ACGT", "AGT"),
            ("ACCCGT", "AGT"),
            ("A", "C"),
            ("ACGTACGT", "TGCATGCA"),
            ("AAAACCCC", "CCCCAAAA"),
            ("ACGTACGTAC", "ACGTTTTACGTAC"),
        ],
    )
    @pytest.mark.parametrize("gap_open,gap_extend", [(2.0, 1.0), (10.0, 0.5), (3.0, 3.0)])
    def test_the_counts_reproduce_the_score(
        self, first: str, second: str, gap_open: float, gap_extend: float
    ) -> None:
        """Substitutions less the charge the counts imply must be the score.

        The charge is one opening per run and one extension per further gap
        column, so this is the arithmetic the score was built from, recomputed
        from nothing but the reported counts.
        """
        matrix = _simple()
        gaps = Gaps(gap_open, gap_extend)
        for mode in ("global", "glocal", "overlap", "local"):
            alignment = align(first, second, matrix, gaps, mode=mode)
            if alignment.empty:
                continue
            metrics = measure(
                alignment, matrix, DNA, len(first), len(second)
            )
            substitutions = sum(
                matrix.score(one, two)
                for one, two in zip(
                    alignment.aligned_first, alignment.aligned_second
                )
                if one != "-" and two != "-"
            )
            charge = (
                metrics.gap_runs * gaps.open
                + (metrics.gaps - metrics.gap_runs) * gaps.extend
            )
            assert substitutions - charge == pytest.approx(alignment.score), (
                mode,
                alignment.aligned_first,
                alignment.aligned_second,
            )

    def test_the_longest_run_is_no_longer_than_the_gap_count(self) -> None:
        matrix = _simple()
        for first, second in (("ACGT", "AGT"), ("A", "C"), ("ACCCGT", "AGT")):
            alignment = align(first, second, matrix, Gaps(2.0, 1.0))
            metrics = measure(alignment, matrix, DNA, len(first), len(second))
            assert metrics.longest_gap <= metrics.gaps
            assert metrics.gap_runs <= metrics.gaps


class TestDenominators:
    def test_identity_is_reported_over_two_denominators(self, simple_dna) -> None:
        """A gappy alignment's two identities differ, which is the point.

        'ACGT' against 'AT' aligns over four columns with two identities, which
        is half the alignment and all of the shorter sequence.
        """
        _alignment, metrics = _measure("ACGT", "AT", simple_dna, DNA)
        assert metrics.length == 4
        assert metrics.identities == 2
        assert metrics.identity == pytest.approx(0.5)
        assert metrics.identity_of_shorter == pytest.approx(1.0)

    def test_the_shorter_sequence_is_the_shorter_whole_one(self, simple_dna) -> None:
        _alignment, metrics = _measure("ACGTACGT", "ACGT", simple_dna, DNA)
        assert metrics.shorter == 4

    def test_a_share_of_nothing_is_absent_rather_than_zero(self) -> None:
        """An empty alignment has no identity, and 0% would be a claim.

        Reporting zero would say the residues disagree and reporting one that
        they agree; neither was looked at.
        """
        metrics = Metrics(
            length=0,
            identities=0,
            similarities=0,
            mismatches=0,
            gaps=0,
            gap_runs=0,
            longest_gap=0,
            gaps_first=0,
            gaps_second=0,
            ambiguous_identities=0,
            identical_unscored=0,
            length_first=0,
            length_second=0,
            unaligned_first=0,
            unaligned_second=0,
        )
        assert metrics.identity is None
        assert metrics.identity_of_shorter is None
        assert metrics.similarity is None
        assert metrics.gap_share is None


class TestSimilarity:
    def test_a_positive_scoring_mismatch_is_similar(self, blosum62) -> None:
        """Leucine against isoleucine scores 2 under BLOSUM62.

        It is similar and not identical, which is the distinction the two
        counts exist to make.
        """
        assert blosum62.score("L", "I") > 0
        metrics = measure(_fake("L", "I"), blosum62, PROTEIN, 1, 1)
        assert metrics.similarities == 1
        assert metrics.identities == 0
        assert metrics.mismatches == 1

    def test_a_negative_scoring_mismatch_is_not_similar(self, blosum62) -> None:
        assert blosum62.score("W", "D") < 0
        metrics = measure(_fake("W", "D"), blosum62, PROTEIN, 1, 1)
        assert metrics.similarities == 0

    def test_an_identical_pair_scoring_below_zero_is_not_similar(self, nuc44) -> None:
        """S against S scores -1 under NUC.4.4, so identity is not a subset.

        This is counted and reported because a reader who assumes every
        identical column is also a similar one would find the two columns
        disagreeing and distrust both.
        """
        metrics = measure(_fake("S", "S"), nuc44, DNA, 1, 1)
        assert metrics.identities == 1
        assert metrics.similarities == 0
        assert metrics.identical_unscored == 1

    def test_an_ordinary_identical_pair_is_both(self, simple_dna) -> None:
        metrics = measure(_fake("A", "A"), simple_dna, DNA, 1, 1)
        assert metrics.identities == 1
        assert metrics.similarities == 1
        assert metrics.identical_unscored == 0


class TestCase:
    """Case is soft-masking, not a different residue.

    The matrix scores 'a' against 'A' as a match, so counting it as a mismatch
    would put the identity count at odds with the score the same column
    contributed to.
    """

    def test_lowercase_against_uppercase_is_identical(self) -> None:
        alignment, metrics = _measure("acgt", "ACGT", _simple(), DNA)
        assert alignment.score == 4.0
        assert metrics.identities == 4
        assert metrics.mismatches == 0

    def test_a_soft_masked_region_still_counts(self) -> None:
        _alignment, metrics = _measure("ACgtACGT", "ACGTACGT", _simple(), DNA)
        assert metrics.identities == 8

    def test_a_real_mismatch_is_still_a_mismatch(self) -> None:
        _alignment, metrics = _measure("acgt", "ACGA", _simple(), DNA)
        assert metrics.identities == 3
        assert metrics.mismatches == 1

    def test_the_marker_line_agrees(self) -> None:
        from align_pair.pairview import markers

        alignment = align("acgt", "ACGT", _simple(), Gaps(2.0, 1.0))
        assert markers(alignment, _simple()) == "||||"

    def test_the_rows_keep_the_case_they_were_given(self) -> None:
        """Masking is worth seeing, so the view is faithful to the input."""
        alignment = align("acgt", "ACGT", _simple(), Gaps(2.0, 1.0))
        assert alignment.aligned_first == "acgt"
        assert alignment.aligned_second == "ACGT"


class TestAmbiguity:
    def test_two_ambiguity_codes_agreeing_are_counted(self, simple_dna) -> None:
        metrics = measure(_fake("ANA", "ANA"), simple_dna, DNA, 3, 3)
        assert metrics.identities == 3
        assert metrics.ambiguous_identities == 1

    def test_plain_residues_are_not_counted_as_ambiguous(self, simple_dna) -> None:
        metrics = measure(_fake("ACGT", "ACGT"), simple_dna, DNA, 4, 4)
        assert metrics.ambiguous_identities == 0

    def test_a_mismatch_with_an_ambiguity_code_is_not_counted(
        self, simple_dna
    ) -> None:
        metrics = measure(_fake("N", "A"), simple_dna, DNA, 1, 1)
        assert metrics.ambiguous_identities == 0


class TestUnaligned:
    def test_a_global_alignment_leaves_nothing_unaligned(self, simple_dna) -> None:
        _alignment, metrics = _measure("ACGT", "AGT", simple_dna, DNA)
        assert metrics.unaligned_first == 0
        assert metrics.unaligned_second == 0

    def test_a_local_alignment_counts_what_it_left_out(self, simple_dna) -> None:
        _alignment, metrics = _measure(
            "TTTACGTTTT", "AAAACGTAAA", simple_dna, DNA, mode="local"
        )
        assert metrics.unaligned_first == 6
        assert metrics.unaligned_second == 6

    def test_an_overlap_counts_both_overhangs(self, simple_dna) -> None:
        _alignment, metrics = _measure(
            "TTTTACGT", "ACGTAAAA", simple_dna, DNA, mode="overlap"
        )
        assert metrics.unaligned_first == 4
        assert metrics.unaligned_second == 4


def _simple():
    """A match/mismatch matrix, for arithmetic that is easy to check by hand."""
    return matrices.simple(1.0, -1.0, "nucleotide")


def _fake(first: str, second: str):
    """An alignment object standing for two rows written out by hand."""
    from align_pair.aligner import Alignment

    return Alignment(
        mode="global",
        score=0.0,
        aligned_first=first,
        aligned_second=second,
        start_first=1,
        end_first=len(first.replace("-", "")),
        start_second=1,
        end_second=len(second.replace("-", "")),
        co_optimal=1,
        co_optimal_capped=False,
        cells=0,
    )
