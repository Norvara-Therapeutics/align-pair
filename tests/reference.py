"""A naive aligner that shares no code with the package, for differential tests.

Every alignment of two short sequences is enumerated directly from the
definition, scored column by column, and the best score and the number of
alignments achieving it are reported. Nothing here is efficient and nothing
here is imported by the package: the point is that it is obviously right, so a
disagreement with ``align_pair.aligner`` is a fault in the dynamic programme and
not in an assumption the two share.

An alignment is a leading overhang, a sequence of columns, and a trailing
overhang. An overhang is residues of one sequence that hang off the end and are
not aligned to anything; the modes differ in which overhangs they allow, and an
overhang is never scored and never shown. The columns are scored in full, each
run of gaps costing ``open + (length - 1) * extend``.

Three rules keep one alignment from being counted twice. An overhang lies in
one sequence, not both at once. The columns do not end with a gap in the
sequence whose trailing overhang is allowed, since that gap and the overhang
beside it are the same run of gaps written two ways. And at least one residue
of each sequence is accounted for, by an overhang or a column, or nothing has
been aligned to anything at all.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, Sequence

#: A column of an alignment: a pair, a gap in the second, a gap in the first.
PAIR = "P"
GAP_SECOND = "X"
GAP_FIRST = "Y"

#: Which sequences may overhang the ends, per mode: the first, the second.
OVERHANGS: dict[str, tuple[bool, bool]] = {
    "global": (False, False),
    "glocal": (False, True),
    "overlap": (True, True),
}


@lru_cache(maxsize=None)
def columns_of(first: int, second: int) -> tuple[tuple[str, ...], ...]:
    """Every sequence of columns consuming these many residues of each.

    A column either consumes one residue of each sequence, one of the first, or
    one of the second, so this is every word over the three column types with
    the right number of each.
    """
    if first == 0 and second == 0:
        return ((),)
    found: list[tuple[str, ...]] = []
    if first and second:
        found.extend((PAIR, *rest) for rest in columns_of(first - 1, second - 1))
    if first:
        found.extend((GAP_SECOND, *rest) for rest in columns_of(first - 1, second))
    if second:
        found.extend((GAP_FIRST, *rest) for rest in columns_of(first, second - 1))
    return tuple(found)


def runs(shape: Sequence[str]) -> list[tuple[str, int]]:
    """The columns as consecutive runs of one type."""
    grouped: list[tuple[str, int]] = []
    for column in shape:
        if grouped and grouped[-1][0] == column:
            grouped[-1] = (column, grouped[-1][1] + 1)
        else:
            grouped.append((column, 1))
    return grouped


def paid_score(
    shape: Sequence[str],
    first: str,
    second: str,
    scores: dict[tuple[str, str], float],
    gap_open: float,
    gap_extend: float,
    row: int = 0,
    column: int = 0,
) -> float:
    """Score columns with every gap paid for, run by run.

    Args:
        shape: the column types, in order.
        first: the whole first sequence.
        second: the whole second sequence.
        scores: residue pair to its score.
        gap_open: what the first residue of a gap costs.
        gap_extend: what each further residue of a gap costs.
        row: how many residues of the first sequence precede these columns.
        column: how many residues of the second sequence precede them.
    """
    total = 0.0
    for kind, length in runs(shape):
        if kind == PAIR:
            for offset in range(length):
                total += scores[(first[row + offset], second[column + offset])]
            row += length
            column += length
            continue
        total -= gap_open + (length - 1) * gap_extend
        if kind == GAP_SECOND:
            row += length
        else:
            column += length
    return total


def _leads(mode: str, first: int, second: int) -> list[tuple[int, int]]:
    """Every leading overhang the mode allows, as residues consumed."""
    overhang_first, overhang_second = OVERHANGS[mode]
    found = [(0, 0)]
    if overhang_first:
        found.extend((row, 0) for row in range(1, first + 1))
    if overhang_second:
        found.extend((0, column) for column in range(1, second + 1))
    return found


def best(
    first: str,
    second: str,
    scores: dict[tuple[str, str], float],
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
    mode: str = "global",
) -> tuple[float, int]:
    """The optimal score and how many alignments achieve it.

    Raises:
        ValueError: if the mode is not one this reference implements.
    """
    if mode == "local":
        return _best_local(first, second, scores, gap_open, gap_extend)
    if mode not in OVERHANGS:
        raise ValueError(f"unknown mode: {mode}")

    length_first = len(first)
    length_second = len(second)
    overhang_first, overhang_second = OVERHANGS[mode]
    # With one sequence empty there is exactly one alignment, all of it gaps.
    # It is free only if the mode lets the sequence that is there overhang the
    # end: a glocal alignment against an empty second sequence still has to
    # account for the first one, which is the point of the mode.
    if not (length_first and length_second):
        if length_first and overhang_first:
            return 0.0, 1
        if length_second and overhang_second:
            return 0.0, 1
        shape = (GAP_SECOND,) * length_first + (GAP_FIRST,) * length_second
        return paid_score(shape, first, second, scores, gap_open, gap_extend), 1

    top = -float("inf")
    count = 0
    for start_row, start_column in _leads(mode, length_first, length_second):
        for end_row in range(start_row, length_first + 1):
            for end_column in range(start_column, length_second + 1):
                trailing_first = length_first - end_row
                trailing_second = length_second - end_column
                if trailing_first and not overhang_first:
                    continue
                if trailing_second and not overhang_second:
                    continue
                if trailing_first and trailing_second:
                    continue
                if not end_row or not end_column:
                    continue
                for shape in columns_of(
                    end_row - start_row, end_column - start_column
                ):
                    if not shape:
                        continue
                    # A gap at either end that the overhang beside it could
                    # have absorbed is that overhang, written out. Counting
                    # both would count one alignment twice.
                    if start_row == 0 and overhang_second:
                        if shape[0] == GAP_FIRST:
                            continue
                    if start_column == 0 and overhang_first:
                        if shape[0] == GAP_SECOND:
                            continue
                    if not trailing_first and overhang_second:
                        if shape[-1] == GAP_FIRST:
                            continue
                    elif trailing_first and overhang_first:
                        if shape[-1] == GAP_SECOND:
                            continue
                    value = paid_score(
                        shape,
                        first,
                        second,
                        scores,
                        gap_open,
                        gap_extend,
                        start_row,
                        start_column,
                    )
                    if value > top:
                        top = value
                        count = 1
                    elif value == top:
                        count += 1
    return top, count


def _best_local(
    first: str,
    second: str,
    scores: dict[tuple[str, str], float],
    gap_open: float,
    gap_extend: float,
) -> tuple[float, int]:
    """The best local alignment, over every pair of subsequences.

    A local alignment begins and ends with an aligned pair, which is the
    convention Smith-Waterman is read under: whenever a gap costs anything, a
    leading or trailing gap can be trimmed for a better score, so an optimal
    local alignment never has one.
    """
    top = -float("inf")
    count = 0
    for start_row in range(len(first)):
        for stop_row in range(start_row + 1, len(first) + 1):
            for start_column in range(len(second)):
                for stop_column in range(start_column + 1, len(second) + 1):
                    for shape in columns_of(
                        stop_row - start_row, stop_column - start_column
                    ):
                        if shape[0] != PAIR or shape[-1] != PAIR:
                            continue
                        value = paid_score(
                            shape,
                            first,
                            second,
                            scores,
                            gap_open,
                            gap_extend,
                            start_row,
                            start_column,
                        )
                        if value > top:
                            top = value
                            count = 1
                        elif value == top:
                            count += 1
    if top <= 0:
        # No pair of subsequences scores ABOVE zero, so the empty alignment is
        # the optimum. A pair scoring exactly zero is that same alignment
        # written out rather than a second one tying with it.
        return 0.0, 1
    return top, count


def table(letters: str, match: float, mismatch: float) -> dict[tuple[str, str], float]:
    """A match/mismatch scoring table over an alphabet."""
    return {
        (one, two): match if one == two else mismatch
        for one in letters
        for two in letters
    }


def from_matrix(
    labels: str, rows: Sequence[Sequence[float]]
) -> dict[tuple[str, str], float]:
    """A scoring table read from a matrix's labels and rows."""
    return {
        (first, second): rows[i][j]
        for i, first in enumerate(labels)
        for j, second in enumerate(labels)
    }


def sequences(letters: str, length: int) -> Iterator[str]:
    """Every sequence of a given length over an alphabet."""
    if length == 0:
        yield ""
        return
    for rest in sequences(letters, length - 1):
        for letter in letters:
            yield letter + rest
