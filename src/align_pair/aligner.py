"""The pairwise dynamic programme, with affine gaps.

One fill serves every alignment mode. Three layers are carried - a pair of
residues aligned, a gap in the second sequence, a gap in the first - which is
what makes a gap of length L cost ``open + (L - 1) * extend`` rather than L
times a flat penalty, after Gotoh (1982). The modes differ only in how the
first row and column are initialised and where the traceback is allowed to
end, so there is one recurrence to get right rather than four.

Positions are 1-based and inclusive on both sequences. Gap penalties are given
as positive numbers and subtracted.

Two things are reported that an aligner usually leaves implicit. The first is
how many alignments share the optimal score: an optimal alignment is rarely
unique, and a reader comparing two tools, or re-running with the sequences
swapped, is entitled to know whether the one shown was chosen from a hundred
equals. The second is whether a banded run's optimal path touched the edge of
its band, in which case the score is a lower bound on the true optimum rather
than the optimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from align_pair.alphabet import GAP
from align_pair.matrices import Matrix

#: The alignment modes, in the order they are listed.
MODES: tuple[str, ...] = ("global", "glocal", "overlap", "local")

#: What each mode leaves unpenalised, for output to quote.
MODE_SUMMARY: dict[str, str] = {
    "global": (
        "both sequences aligned end to end, every gap penalised "
        "(Needleman-Wunsch)"
    ),
    "glocal": (
        "the first sequence aligned end to end within the second; gaps at the "
        "ends of the second are free"
    ),
    "overlap": (
        "gaps at either end of either sequence are free, so a suffix of one "
        "may align to a prefix of the other"
    ),
    "local": (
        "the best-scoring pair of subsequences, which begins and ends with an "
        "aligned pair (Smith-Waterman)"
    ),
}

#: Counting stops here. A pair of sequences can admit more co-optimal
#: alignments than there are atoms to write them with, and the exact figure
#: past this point tells a reader nothing the bound does not.
CO_OPTIMAL_CAP: int = 10**12

#: Traceback bits. A cell records every predecessor that achieves its score,
#: not only the one the traceback takes, which is what makes the co-optimal
#: count exact.
_FROM_M: int = 1
_FROM_X: int = 2
_FROM_Y: int = 4
_FROM_START: int = 8

#: Which predecessor the traceback prefers, per layer. A diagonal step is
#: preferred to a gap, continuing a gap to opening one, and continuing an
#: alignment to restarting it.
#:
#: This is a local preference applied at each step, not a second objective the
#: programme optimises: it makes the choice deterministic and tends towards
#: fewer, longer gap runs, but it does not minimise them. Where several
#: alignments share the optimal score, one of them may have fewer runs than the
#: one shown - 'AC' against 'CCA' under a match of 1, a mismatch of -1 and a
#: free extension has three co-optimal alignments, and the one these
#: preferences reach has two gap runs where another has one. Minimising runs
#: would need the run count carried through every cell as a tie-break, which is
#: a different programme; the count of co-optimal alignments is reported
#: instead, so a reader knows the shown alignment was chosen from a set.
_PRIORITY: dict[int, tuple[int, ...]] = {
    0: (_FROM_M, _FROM_X, _FROM_Y, _FROM_START),
    1: (_FROM_X, _FROM_M, _FROM_Y),
    2: (_FROM_Y, _FROM_M, _FROM_X),
}

_LAYER_OF = {_FROM_M: 0, _FROM_X: 1, _FROM_Y: 2}

_NEG = -math.inf


class AlignmentError(ValueError):
    """Raised when an alignment cannot be computed as asked."""


@dataclass(frozen=True)
class Gaps:
    """An affine gap penalty.

    A gap of length L costs ``open + (L - 1) * extend``: the first residue of a
    gap costs ``open`` and each further residue ``extend``. This is the
    convention of EMBOSS needle and water, so the defaults here produce the
    same score as those programs given the same matrix. The BLAST family
    charges ``open + L * extend`` instead, so a score compared against blastp
    differs by one extension per gap.

    Attributes:
        open: penalty for the first residue of a gap, a positive number.
        extend: penalty for each further residue of a gap.
    """

    open: float = 10.0
    extend: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (("open", self.open), ("extend", self.extend)):
            if not math.isfinite(value):
                raise AlignmentError(f"gap {name} penalty must be a finite number")
            if value < 0:
                raise AlignmentError(
                    f"gap {name} penalty must not be negative, got {value}; "
                    "penalties are given as positive numbers and subtracted"
                )

    def cost(self, length: int) -> float:
        """What a gap of ``length`` residues costs."""
        if length <= 0:
            return 0.0
        return self.open + (length - 1) * self.extend

    @property
    def free(self) -> bool:
        """Whether gaps cost nothing at all."""
        return self.open == 0 and self.extend == 0


@dataclass(frozen=True)
class Alignment:
    """One optimal alignment of two sequences.

    Attributes:
        mode: the alignment mode this was computed in.
        score: the optimal score.
        aligned_first: the first sequence with gap characters inserted.
        aligned_second: the second sequence with gap characters inserted.
        start_first: 1-based position in the first sequence of its first
            aligned residue, 0 when none of it is aligned.
        end_first: 1-based position of its last aligned residue, 0 when none.
        start_second: the same for the second sequence.
        end_second: the same for the second sequence.
        co_optimal: how many alignments share this score, counted up to
            ``CO_OPTIMAL_CAP``.
        co_optimal_capped: True when counting stopped at the cap, so
            ``co_optimal`` is a lower bound.
        cells: how many cells of the dynamic programme were computed.
        band: the band half-width, or None when the whole matrix was filled.
        band_limited: True when the band did not span the whole matrix, so
            this score is a lower bound on the unbanded optimum rather than
            the optimum. It is not a claim that anything was missed, and its
            being False is the only claim of optimality a banded run can make.
        band_margin: how many diagonals of the band the path left unused on
            its nearer side, or None when the run was not banded. A path that
            stops well short of the edge is unlikely to have been cut off by
            it, but only a wider band settles the question.
    """

    mode: str
    score: float
    aligned_first: str
    aligned_second: str
    start_first: int
    end_first: int
    start_second: int
    end_second: int
    co_optimal: int
    co_optimal_capped: bool
    cells: int
    band: int | None = None
    band_limited: bool = False
    band_margin: int | None = None

    def __len__(self) -> int:
        return len(self.aligned_first)

    @property
    def length(self) -> int:
        """How many columns the alignment has."""
        return len(self.aligned_first)

    @property
    def columns(self) -> list[tuple[str, str]]:
        """The alignment as a list of residue pairs."""
        return list(zip(self.aligned_first, self.aligned_second))

    @property
    def empty(self) -> bool:
        """Whether no residues were aligned at all."""
        return not self.aligned_first


def band_diagonals(first: int, second: int, band: int) -> tuple[int, int]:
    """The lowest and highest diagonal a banded global path may use.

    A global path starts on diagonal 0 and ends on diagonal ``second - first``,
    so the band has to contain both whatever its width: it runs from ``band``
    below the lower of the two to ``band`` above the higher. A band narrower
    than the length difference is therefore widened to hold a complete path
    rather than refused.
    """
    difference = second - first
    return min(0, difference) - band, max(0, difference) + band


def _ranges(
    first: int, second: int, band: int | None
) -> tuple[list[tuple[int, int]], int]:
    """Each row's range of columns, and the total number of cells."""
    if band is None:
        return [(0, second) for _ in range(first + 1)], (first + 1) * (second + 1)
    low, high = band_diagonals(first, second, band)
    rows: list[tuple[int, int]] = []
    total = 0
    for row in range(first + 1):
        start = max(0, row + low)
        stop = min(second, row + high)
        rows.append((start, stop))
        total += stop - start + 1
    return rows, total


def cells(first: int, second: int, band: int | None = None) -> int:
    """How many cells a run over these lengths would compute.

    This is the measure of work, and so of time. It is not the measure of
    memory: see ``footprint``, which a narrow band leaves large.
    """
    return _ranges(first, second, band)[1]


#: Bytes a cell of traceback costs: one byte in each of the three mask arrays.
MASK_BYTES: int = 3

#: Bytes a column of the dynamic programme costs, whatever the band. Twelve
#: arrays are held at full width - a score and a path count for each of the
#: three layers, in both the current row and the previous one - and each entry
#: costs a pointer plus the boxed number it points at. Measured at about 480
#: bytes a column across alignments from 3,000 to 160,000 residues, which is
#: the figure this uses; see ``footprint``.
COLUMN_BYTES: int = 480


def footprint(first: int, second: int, band: int | None = None) -> int:
    """About how many bytes a run over these lengths would hold.

    A band reduces the cells and so the traceback, but not the rows: those span
    the second sequence whatever the band, so a narrow band over a long pair
    still costs hundreds of bytes per residue of it. Counting cells alone would
    put a 40,000-residue band-zero run at 40,001 cells and 120 kB when it in
    fact holds about 19 MB, which is why the limit is on this and not on cells.

    The figure is an estimate from measured allocation, not an exact
    accounting, so it is used to refuse work that is clearly too large rather
    than to predict the last megabyte.
    """
    return MASK_BYTES * cells(first, second, band) + COLUMN_BYTES * (second + 1)


def align(
    first: str,
    second: str,
    matrix: Matrix,
    gaps: Gaps | None = None,
    mode: str = "global",
    band: int | None = None,
) -> Alignment:
    """Align two sequences optimally.

    Args:
        first: the first sequence, which the alignment's upper row shows.
        second: the second sequence, shown in the lower row.
        matrix: the substitution matrix to score residue pairs with.
        gaps: the affine gap penalty; the EMBOSS defaults when omitted.
        mode: one of ``MODES``.
        band: restrict the fill to this many diagonals either side of the main
            one. Global mode only, since no other mode has a fixed end point
            for a band to be drawn around.

    Raises:
        AlignmentError: if the mode is unknown, if a band is asked for in a
            mode that has no band, or if the band is negative.
        MatrixError: if the matrix cannot score a residue of either sequence.
    """
    if mode not in MODES:
        known = ", ".join(MODES)
        raise AlignmentError(f"unknown alignment mode: {mode!r}; choose from {known}")
    if band is not None:
        if mode != "global":
            raise AlignmentError(
                f"a band is only defined for global alignment, not for {mode}: "
                "the other modes have no fixed end point to draw a band around"
            )
        if band < 0:
            raise AlignmentError(f"band must not be negative, got {band}")
    gaps = gaps or Gaps()

    rows = [matrix.row(residue) for residue in first]
    columns = matrix.columns(second)
    length_first = len(first)
    length_second = len(second)
    spans, total = _ranges(length_first, length_second, band)

    state = _fill(rows, columns, gaps, mode, spans, total)
    end_layer, end_row, end_column, score = _end_state(
        state, mode, length_first, length_second, spans
    )
    count, capped = _co_optimal(state, mode, length_first, length_second, spans, score)

    if score == _NEG:
        raise AlignmentError(
            "no alignment is possible under these penalties, which should not "
            "happen; please report the sequences that produced it"
        )

    aligned_first, aligned_second, start_row, start_column, low, high = _traceback(
        first, second, state, spans, end_layer, end_row, end_column, mode
    )

    limited = False
    margin: int | None = None
    if band is not None:
        band_low, band_high = band_diagonals(length_first, length_second, band)
        covers_all = band_low <= -length_first and band_high >= length_second
        # A path that never reaches the edge of its band is not thereby
        # optimal: the better path may lie outside the band without this one
        # going anywhere near it. So the band is reported as limiting whenever
        # it leaves any of the matrix out, and how much room the path had is
        # reported separately, as the diagnostic it is rather than as proof.
        limited = not covers_all
        margin = min(low - band_low, band_high - high)

    aligned = bool(aligned_first)
    return Alignment(
        mode=mode,
        score=0.0 if (mode == "local" and not aligned) else score,
        aligned_first=aligned_first,
        aligned_second=aligned_second,
        start_first=start_row + 1 if aligned and end_row > start_row else 0,
        end_first=end_row if aligned and end_row > start_row else 0,
        start_second=start_column + 1 if aligned and end_column > start_column else 0,
        end_second=end_column if aligned and end_column > start_column else 0,
        co_optimal=count,
        co_optimal_capped=capped,
        cells=total,
        band=band,
        band_limited=limited,
        band_margin=margin,
    )


@dataclass
class _State:
    """Everything the fill leaves behind for the traceback and the count."""

    masks: tuple[bytearray, bytearray, bytearray]
    offsets: list[int]
    last_values: tuple[list[float], list[float], list[float]]
    last_counts: tuple[list[int], list[int], list[int]]
    edge_values: list[tuple[float, float, float]]
    edge_counts: list[tuple[int, int, int]]
    best_local: tuple[float, int, int, int]


def _fill(
    rows: Sequence[tuple[float, ...]],
    columns: Sequence[int],
    gaps: Gaps,
    mode: str,
    spans: Sequence[tuple[int, int]],
    total: int,
) -> _State:
    """Compute the three layers, recording every optimal predecessor.

    The path counts are accumulated in the same pass: a state's count is the
    sum of its optimal predecessors' counts, which is only cheap while the
    predecessors' counts are still in the previous row.
    """
    length_first = len(rows)
    length_second = len(columns)
    gap_open = gaps.open
    gap_extend = gaps.extend
    local = mode == "local"
    free_first = mode == "overlap"
    free_second = mode in ("glocal", "overlap")
    cap = CO_OPTIMAL_CAP

    mask_m = bytearray(total)
    mask_x = bytearray(total)
    mask_y = bytearray(total)
    offsets: list[int] = []
    position = 0
    for start, stop in spans:
        offsets.append(position - start)
        position += stop - start + 1

    width = length_second + 1
    prev_m = [_NEG] * width
    prev_x = [_NEG] * width
    prev_y = [_NEG] * width
    count_m = [0] * width
    count_x = [0] * width
    count_y = [0] * width

    start, stop = spans[0]
    base = offsets[0]
    if not local:
        # The corner is the state an alignment of two empty prefixes is in. A
        # local alignment has no such state: beginning is expressed by the
        # start bit, and counting the corner as well would count every local
        # alignment twice, once as itself and once as an extension of nothing.
        prev_m[0] = 0.0
        count_m[0] = 1
        for column in range(max(1, start), stop + 1):
            prev_y[column] = 0.0 if free_second else -gaps.cost(column)
            count_y[column] = 1
            mask_y[base + column] = _FROM_Y if column >= 2 else _FROM_M

    edge_values: list[tuple[float, float, float]] = []
    edge_counts: list[tuple[int, int, int]] = []
    if stop == length_second:
        edge_values.append((prev_m[length_second], prev_x[length_second],
                            prev_y[length_second]))
        edge_counts.append((count_m[length_second], count_x[length_second],
                            count_y[length_second]))
    else:
        edge_values.append((_NEG, _NEG, _NEG))
        edge_counts.append((0, 0, 0))

    best_value = _NEG
    best_row = 0
    best_column = 0
    best_count = 0

    current_m = [_NEG] * width
    current_x = [_NEG] * width
    current_y = [_NEG] * width
    current_count_m = [0] * width
    current_count_x = [0] * width
    current_count_y = [0] * width

    for row in range(1, length_first + 1):
        start, stop = spans[row]
        base = offsets[row]
        scores = rows[row - 1]
        # Only the span this row reads and writes is cleared, which keeps a
        # narrow band narrow. Two cells either side of the span are cleared as
        # well: the one before it is read as the left-hand neighbour, and the
        # one after it is read by the row below, whose span may reach one
        # column further. These arrays are two rows old after the swap, so a
        # cell left uncleared holds a value from the wrong row rather than
        # nothing.
        blank = max(0, start - 1)
        edge = min(stop + 1, length_second)
        span = [_NEG] * (edge - blank + 1)
        zeros = [0] * (edge - blank + 1)
        current_m[blank : edge + 1] = span
        current_x[blank : edge + 1] = span
        current_y[blank : edge + 1] = span
        current_count_m[blank : edge + 1] = zeros
        current_count_x[blank : edge + 1] = zeros
        current_count_y[blank : edge + 1] = zeros

        first_column = start
        if start == 0:
            if not local:
                current_x[0] = 0.0 if free_first else -gaps.cost(row)
                current_count_x[0] = 1
                mask_x[base] = _FROM_X if row >= 2 else _FROM_M
            first_column = 1

        for column in range(first_column, stop + 1):
            left = column - 1
            # The layer for an aligned pair: whatever came before, diagonally.
            value = prev_m[left]
            mask = _FROM_M
            other = prev_x[left]
            if other > value:
                value = other
                mask = _FROM_X
            elif other == value:
                mask |= _FROM_X
            other = prev_y[left]
            if other > value:
                value = other
                mask = _FROM_Y
            elif other == value:
                mask |= _FROM_Y
            if local:
                if value < 0.0:
                    value = 0.0
                    mask = _FROM_START
                elif value == 0.0:
                    mask |= _FROM_START
            if value == _NEG:
                mask = 0
                total_count = 0
            else:
                total_count = 0
                if mask & _FROM_M:
                    total_count += count_m[left]
                if mask & _FROM_X:
                    total_count += count_x[left]
                if mask & _FROM_Y:
                    total_count += count_y[left]
                if mask & _FROM_START:
                    total_count += 1
                if total_count > cap:
                    total_count = cap
                value += scores[columns[column - 1]]
            current_m[column] = value
            current_count_m[column] = total_count
            mask_m[base + column] = mask

            if local and value > best_value:
                best_value = value
                best_row = row
                best_column = column
                best_count = total_count
            elif local and value == best_value and value != _NEG:
                best_count += total_count
                if best_count > cap:
                    best_count = cap

            # A gap in the second sequence: a residue of the first consumed.
            value = prev_x[column] - gap_extend
            mask = _FROM_X
            other = prev_m[column] - gap_open
            if other > value:
                value = other
                mask = _FROM_M
            elif other == value:
                mask |= _FROM_M
            other = prev_y[column] - gap_open
            if other > value:
                value = other
                mask = _FROM_Y
            elif other == value:
                mask |= _FROM_Y
            if value == _NEG:
                mask = 0
                total_count = 0
            else:
                total_count = 0
                if mask & _FROM_M:
                    total_count += count_m[column]
                if mask & _FROM_X:
                    total_count += count_x[column]
                if mask & _FROM_Y:
                    total_count += count_y[column]
                if total_count > cap:
                    total_count = cap
            current_x[column] = value
            current_count_x[column] = total_count
            mask_x[base + column] = mask

            # A gap in the first sequence: a residue of the second consumed.
            value = current_y[left] - gap_extend
            mask = _FROM_Y
            other = current_m[left] - gap_open
            if other > value:
                value = other
                mask = _FROM_M
            elif other == value:
                mask |= _FROM_M
            other = current_x[left] - gap_open
            if other > value:
                value = other
                mask = _FROM_X
            elif other == value:
                mask |= _FROM_X
            if value == _NEG:
                mask = 0
                total_count = 0
            else:
                total_count = 0
                if mask & _FROM_M:
                    total_count += current_count_m[left]
                if mask & _FROM_X:
                    total_count += current_count_x[left]
                if mask & _FROM_Y:
                    total_count += current_count_y[left]
                if total_count > cap:
                    total_count = cap
            current_y[column] = value
            current_count_y[column] = total_count
            mask_y[base + column] = mask

        if stop == length_second:
            edge_values.append(
                (
                    current_m[length_second],
                    current_x[length_second],
                    current_y[length_second],
                )
            )
            edge_counts.append(
                (
                    current_count_m[length_second],
                    current_count_x[length_second],
                    current_count_y[length_second],
                )
            )
        else:
            edge_values.append((_NEG, _NEG, _NEG))
            edge_counts.append((0, 0, 0))

        prev_m, current_m = current_m, prev_m
        prev_x, current_x = current_x, prev_x
        prev_y, current_y = current_y, prev_y
        count_m, current_count_m = current_count_m, count_m
        count_x, current_count_x = current_count_x, count_x
        count_y, current_count_y = current_count_y, count_y

    return _State(
        masks=(mask_m, mask_x, mask_y),
        offsets=offsets,
        last_values=(prev_m, prev_x, prev_y),
        last_counts=(count_m, count_x, count_y),
        edge_values=edge_values,
        edge_counts=edge_counts,
        best_local=(best_value, best_row, best_column, best_count),
    )


def _end_candidates(
    state: _State,
    mode: str,
    length_first: int,
    length_second: int,
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int, int, float, int]]:
    """Every state an optimal path may end in, in the order ties are broken.

    Ties are broken towards the alignment that leaves fewest residues
    unaligned, and then towards an aligned pair rather than a gap.
    """
    candidates: list[tuple[int, int, int, float, int]] = []
    values = state.last_values
    counts = state.last_counts
    start, stop = spans[length_first]

    if mode == "local":
        value, row, column, count = state.best_local
        if value == _NEG or value <= 0.0:
            # No pair of subsequences scores ABOVE zero, so the optimum is the
            # empty alignment, which scores zero. A pair scoring exactly zero is
            # the empty alignment written out, not a second alignment that ties
            # with it. The count then excludes it: a reported alignment is never
            # the empty one, and the empty alignment is never counted beside a
            # reported one.
            return []
        return [(0, row, column, value, count)]

    if mode == "global" or not (length_first and length_second):
        # With one sequence empty there is one alignment, all of it gaps, and
        # it is the alignment whatever the mode says about end gaps. Offering
        # the free-end states as well would count that one alignment once for
        # every column the free gap could have stopped at.
        for layer in (0, 1, 2):
            candidates.append(
                (
                    layer,
                    length_first,
                    length_second,
                    values[layer][length_second],
                    counts[layer][length_second],
                )
            )
        return candidates

    # A free end gap can consume a whole sequence, so a state in the first row
    # or the first column stands for an alignment with nothing aligned to
    # anything. Those are excluded while there is any alternative, which there
    # is whenever both sequences hold a residue.
    floor = 1 if length_first and length_second else 0

    # An alignment does not end with a gap in the sequence whose overhang is
    # free: that gap and the overhang beside it are the same run of gaps, and
    # ending there would report one alignment twice, once with a column shown
    # and once with it dropped. It is never the better choice either, since
    # stopping one column earlier is the same alignment without the penalty.

    # The last row, longest alignment of the second sequence first.
    for column in range(stop, max(start, floor) - 1, -1):
        for layer in (0, 1):
            candidates.append(
                (
                    layer,
                    length_first,
                    column,
                    values[layer][column],
                    counts[layer][column],
                )
            )
    if mode == "overlap":
        # The last column, which the row above has already offered at its end.
        for row in range(length_first - 1, floor - 1, -1):
            for layer in (0, 2):
                candidates.append(
                    (
                        layer,
                        row,
                        length_second,
                        state.edge_values[row][layer],
                        state.edge_counts[row][layer],
                    )
                )
    return candidates


def _end_state(
    state: _State,
    mode: str,
    length_first: int,
    length_second: int,
    spans: Sequence[tuple[int, int]],
) -> tuple[int, int, int, float]:
    """Where the traceback starts, and the optimal score."""
    candidates = _end_candidates(state, mode, length_first, length_second, spans)
    if not candidates:
        return 0, 0, 0, 0.0
    best = max(value for _layer, _row, _column, value, _count in candidates)
    for layer, row, column, value, _count in candidates:
        if value == best:
            return layer, row, column, best
    return 0, 0, 0, best


def _co_optimal(
    state: _State,
    mode: str,
    length_first: int,
    length_second: int,
    spans: Sequence[tuple[int, int]],
    score: float,
) -> tuple[int, bool]:
    """How many alignments share the optimal score.

    Every state records each predecessor that achieves its score, so the count
    of optimal paths into a state is the sum of its optimal predecessors'
    counts. Summing that over every state an optimal path may end in counts
    the co-optimal alignments exactly, up to the cap.
    """
    candidates = _end_candidates(state, mode, length_first, length_second, spans)
    if not candidates:
        return 1, False
    total = 0
    for _layer, _row, _column, value, count in candidates:
        if value == score:
            total += count
    # A count that reached the cap is held AT the cap, and any sum drawing on
    # it is at least the cap and so is capped in turn. The bound therefore
    # travels with the count itself, and a run in which some unrelated state
    # saturated does not make this alignment's count a bound: fifty identical
    # residues under free gaps have one optimal alignment, and reporting 'at
    # least 1' would be false.
    total = min(total, CO_OPTIMAL_CAP)
    return total, total >= CO_OPTIMAL_CAP


def _traceback(
    first: str,
    second: str,
    state: _State,
    spans: Sequence[tuple[int, int]],
    layer: int,
    row: int,
    column: int,
    mode: str,
) -> tuple[str, str, int, int, int, int]:
    """Walk the recorded predecessors back, building the aligned strings.

    A free end gap is not part of the alignment and is not written into it. The
    walk therefore stops on reaching the boundary state the mode makes free -
    the first row for a free overhang of the second sequence, the first column
    for one of the first - rather than emitting the gap columns that would
    reach the corner. Writing them would show an alignment whose own columns do
    not add up to the score beside it.
    """
    masks = state.masks
    offsets = state.offsets
    free_first = mode == "overlap"
    free_second = mode in ("glocal", "overlap")
    upper: list[str] = []
    lower: list[str] = []
    low = column - row
    high = low

    while True:
        diagonal = column - row
        if diagonal < low:
            low = diagonal
        if diagonal > high:
            high = diagonal
        if layer == 2 and row == 0 and free_second:
            break
        if layer == 1 and column == 0 and free_first:
            break
        start, stop = spans[row]
        if not start <= column <= stop:
            break
        mask = masks[layer][offsets[row] + column]
        if layer == 0:
            if row == 0 or column == 0 or mask == 0:
                break
            upper.append(first[row - 1])
            lower.append(second[column - 1])
            row -= 1
            column -= 1
        elif layer == 1:
            if row == 0 or mask == 0:
                break
            upper.append(first[row - 1])
            lower.append(GAP)
            row -= 1
        else:
            if column == 0 or mask == 0:
                break
            upper.append(GAP)
            lower.append(second[column - 1])
            column -= 1
        for bit in _PRIORITY[layer]:
            if mask & bit:
                if bit == _FROM_START:
                    mask = 0
                    break
                layer = _LAYER_OF[bit]
                break
        else:
            break
        if mask == 0:
            break

    upper.reverse()
    lower.reverse()
    return "".join(upper), "".join(lower), row, column, low, high
