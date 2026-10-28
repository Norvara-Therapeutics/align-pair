"""The dynamic programme against a brute-force enumeration of every alignment.

``tests/reference.py`` enumerates alignments directly from the definition and
shares no code with the package, so agreement on both the optimal score and the
number of alignments achieving it is evidence about the recurrence rather than
about a shared assumption. The count is the stronger of the two: a score can be
right while the traceback follows an edge that should not exist, and the count
notices.
"""

from __future__ import annotations

import pytest
import reference

from align_pair import matrices
from align_pair.aligner import MODES, Gaps, align

#: Gap penalties the sweeps run under. The last two are degenerate on purpose:
#: free gaps make every gap placement co-optimal, and an extension dearer than
#: an opening inverts the usual preference for one long gap over two short ones.
PENALTIES: tuple[tuple[float, float], ...] = (
    (2.0, 1.0),
    (2.0, 2.0),
    (5.0, 0.5),
    (10.0, 0.5),
    (1.0, 3.0),
    (0.0, 0.0),
    (3.0, 0.0),
)

_SIMPLE_SCORES = reference.table("ACGTURYSWKMBDHVN", 1.0, -1.0)


def _pairs(letters: str, longest: int):
    for length_first in range(longest + 1):
        for length_second in range(longest + 1):
            for first in reference.sequences(letters, length_first):
                for second in reference.sequences(letters, length_second):
                    yield first, second


def _rescore(alignment, matrix, gaps: Gaps) -> float:
    """Recompute an alignment's score from the columns it reports.

    This is the alignment's own account of itself: a reported alignment whose
    columns do not add up to the score beside it is wrong however the score was
    arrived at.
    """
    total = 0.0
    previous = ""
    for upper, lower in zip(alignment.aligned_first, alignment.aligned_second):
        if upper == "-" or lower == "-":
            kind = "first" if upper == "-" else "second"
            total -= gaps.extend if previous == kind else gaps.open
            previous = kind
            continue
        total += matrix.score(upper, lower)
        previous = ""
    return total


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gap_open,gap_extend", PENALTIES)
def test_every_mode_agrees_with_brute_force_on_two_letters(
    mode: str, gap_open: float, gap_extend: float, simple_dna
) -> None:
    gaps = Gaps(gap_open, gap_extend)
    for first, second in _pairs("AC", 4):
        expected = reference.best(
            first, second, _SIMPLE_SCORES, gap_open, gap_extend, mode
        )
        alignment = align(first, second, simple_dna, gaps, mode=mode)
        assert (alignment.score, alignment.co_optimal) == expected, (
            mode,
            gaps,
            first,
            second,
        )


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_agrees_with_brute_force_on_three_letters(
    mode: str, simple_dna
) -> None:
    gaps = Gaps(2.0, 1.0)
    for first, second in _pairs("ACG", 4):
        expected = reference.best(first, second, _SIMPLE_SCORES, 2.0, 1.0, mode)
        alignment = align(first, second, simple_dna, gaps, mode=mode)
        assert (alignment.score, alignment.co_optimal) == expected, (
            mode,
            first,
            second,
        )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("gap_open,gap_extend", PENALTIES)
def test_a_reported_alignment_adds_up_to_its_own_score(
    mode: str, gap_open: float, gap_extend: float, simple_dna
) -> None:
    gaps = Gaps(gap_open, gap_extend)
    for first, second in _pairs("AC", 4):
        alignment = align(first, second, simple_dna, gaps, mode=mode)
        if alignment.empty:
            assert alignment.score == 0.0
            continue
        assert _rescore(alignment, simple_dna, gaps) == pytest.approx(alignment.score)


@pytest.mark.parametrize("mode", MODES)
def test_reported_coordinates_name_the_residues_shown(mode: str, simple_dna) -> None:
    gaps = Gaps(2.0, 1.0)
    for first, second in _pairs("AC", 4):
        alignment = align(first, second, simple_dna, gaps, mode=mode)
        upper = alignment.aligned_first.replace("-", "")
        lower = alignment.aligned_second.replace("-", "")
        if upper:
            assert first[alignment.start_first - 1 : alignment.end_first] == upper
        else:
            assert (alignment.start_first, alignment.end_first) == (0, 0)
        if lower:
            assert second[alignment.start_second - 1 : alignment.end_second] == lower
        else:
            assert (alignment.start_second, alignment.end_second) == (0, 0)


@pytest.mark.parametrize("mode", ("global", "glocal", "overlap"))
def test_a_published_matrix_agrees_with_brute_force(mode: str) -> None:
    """The sweeps above use match/mismatch, where every pair scores alike.

    A published matrix has an uneven score distribution, so an off-by-one in
    the diagonal lookup shows up here and not there.
    """
    matrix = matrices.load("BLOSUM62")
    scores = reference.from_matrix(matrix.labels, matrix.scores)
    gaps = Gaps(10.0, 0.5)
    for first, second in _pairs("ACW", 3):
        expected = reference.best(first, second, scores, 10.0, 0.5, mode)
        alignment = align(first, second, matrix, gaps, mode=mode)
        assert (alignment.score, alignment.co_optimal) == expected, (
            mode,
            first,
            second,
        )


def test_local_agrees_with_brute_force_under_a_published_matrix() -> None:
    matrix = matrices.load("BLOSUM62")
    scores = reference.from_matrix(matrix.labels, matrix.scores)
    for gap_open, gap_extend in ((10.0, 0.5), (2.0, 2.0)):
        gaps = Gaps(gap_open, gap_extend)
        for first, second in _pairs("ACW", 3):
            expected = reference.best(
                first, second, scores, gap_open, gap_extend, "local"
            )
            alignment = align(first, second, matrix, gaps, mode="local")
            assert (alignment.score, alignment.co_optimal) == expected, (
                gaps,
                first,
                second,
            )


def test_the_comparison_can_fail() -> None:
    """A differential test that cannot fail proves nothing.

    The reference is asked for a score under penalties the package was not
    given, and the two must disagree; if they agreed here, the comparison above
    would be comparing something to itself.
    """
    matrix = matrices.simple(1.0, -1.0, "nucleotide")
    first, second = "ACGTACG", "AGTACG"
    honest = reference.best(first, second, _SIMPLE_SCORES, 2.0, 1.0, "global")
    wrong = reference.best(first, second, _SIMPLE_SCORES, 20.0, 10.0, "global")
    alignment = align(first, second, matrix, Gaps(2.0, 1.0), mode="global")
    assert (alignment.score, alignment.co_optimal) == honest
    assert (alignment.score, alignment.co_optimal) != wrong
