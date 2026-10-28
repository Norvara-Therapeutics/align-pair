"""The tabular outputs, built once and rendered twice.

Every table the tool can write exists here as columns and rows of text. The
``--output`` file, the tables embedded in the HTML report for download and the
rows printed to standard output are all the same function's result, so a table
downloaded from the report is the table the file would have held.

Values are written unrounded: a float is written in the shortest form that
reads back as the same double, never to one decimal place. Rounding belongs to
the human-facing renderings, which call ``text`` for it. A value that does not
exist is written as an empty field rather than as a zero, since an identity of
0% and no identity at all are different facts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from align_pair.analysis import Analysis, Pair
from align_pair.matrices import Matrix
from align_pair.pairview import markers

#: What separates fields. Tab, because a FASTA description may hold a comma.
SEPARATOR: str = "\t"


def field(value: object) -> str:
    """One value, written for a tab-separated file.

    Floats keep full precision, None becomes an empty field, and a tab or
    newline inside a description becomes a space so that one row stays one row.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    text = str(value)
    for character in ("\t", "\r", "\n"):
        text = text.replace(character, " ")
    return text


def render(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A whole table as text, with its header line and a trailing newline."""
    lines = [SEPARATOR.join(columns)]
    lines.extend(SEPARATOR.join(field(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


PAIR_COLUMNS: tuple[str, ...] = (
    "first",
    "second",
    "mode",
    "matrix",
    "strand",
    "score",
    "length",
    "identities",
    "identity_of_alignment",
    "identity_of_shorter",
    "similarities",
    "similarity_of_alignment",
    "mismatches",
    "gaps",
    "gap_of_alignment",
    "gap_runs",
    "longest_gap",
    "gaps_in_first",
    "gaps_in_second",
    "length_first",
    "length_second",
    "start_first",
    "end_first",
    "start_second",
    "end_second",
    "unaligned_first",
    "unaligned_second",
    "co_optimal",
    "co_optimal_at_least",
    "cells",
    "band",
    "band_limited",
    "band_margin",
)


def pair_row(pair: Pair, analysis: Analysis) -> list[object]:
    """One pair as a row of the summary table."""
    alignment = pair.alignment
    metrics = pair.metrics
    start_second, end_second = pair.span_second
    return [
        pair.first.identifier,
        pair.second.identifier,
        alignment.mode,
        analysis.matrix.name,
        pair.strand,
        alignment.score,
        metrics.length,
        metrics.identities,
        metrics.identity,
        metrics.identity_of_shorter,
        metrics.similarities,
        metrics.similarity,
        metrics.mismatches,
        metrics.gaps,
        metrics.gap_share,
        metrics.gap_runs,
        metrics.longest_gap,
        metrics.gaps_first,
        metrics.gaps_second,
        metrics.length_first,
        metrics.length_second,
        alignment.start_first or None,
        alignment.end_first or None,
        start_second or None,
        end_second or None,
        metrics.unaligned_first,
        metrics.unaligned_second,
        alignment.co_optimal,
        alignment.co_optimal_capped,
        alignment.cells,
        alignment.band,
        alignment.band_limited,
        alignment.band_margin,
    ]


def pair_rows(analysis: Analysis) -> list[list[object]]:
    """Every pair of a run as rows of the summary table."""
    return [pair_row(pair, analysis) for pair in analysis.pairs]


COLUMN_COLUMNS: tuple[str, ...] = (
    "first",
    "second",
    "strand",
    "column",
    "residue_first",
    "residue_second",
    "position_first",
    "position_second",
    "score",
    "marker",
)


def column_rows(pair: Pair, matrix: Matrix) -> list[list[object]]:
    """Every column of one alignment as a row.

    Positions are positions in the records, so on the minus strand the second
    position counts down: the residue shown is the record's, read the other
    way, and a reader checking a column against the file needs the number the
    file uses.
    """
    alignment = pair.alignment
    marks = markers(alignment, matrix)
    position_first = max(alignment.start_first - 1, 0)
    position_second = max(alignment.start_second - 1, 0)
    length_second = pair.metrics.length_second if pair.strand == "minus" else None

    def place(position: int) -> int:
        if length_second is None:
            return position
        return length_second - position + 1

    rows: list[list[object]] = []
    for index, (upper, lower) in enumerate(
        zip(alignment.aligned_first, alignment.aligned_second), start=1
    ):
        score: float | None = None
        if upper != "-" and lower != "-":
            score = matrix.score(upper, lower)
        if upper != "-":
            position_first += 1
        if lower != "-":
            position_second += 1
        rows.append(
            [
                pair.first.identifier,
                pair.second.identifier,
                pair.strand,
                index,
                upper,
                lower,
                position_first if upper != "-" else None,
                place(position_second) if lower != "-" else None,
                score,
                marks[index - 1],
            ]
        )
    return rows


def all_column_rows(analysis: Analysis) -> list[list[object]]:
    """Every column of every pair in a run."""
    rows: list[list[object]] = []
    for pair in analysis.pairs:
        rows.extend(column_rows(pair, analysis.matrix))
    return rows


MATRIX_COLUMNS: tuple[str, ...] = ("name", "scores", "residues", "range", "summary")


def matrix_rows(listing: Sequence[tuple[str, str, str]]) -> list[list[object]]:
    """The built-in matrices as rows, from ``matrices.listing``."""
    from align_pair import matrices as tables

    rows: list[list[object]] = []
    for name, kind, summary in listing:
        matrix = tables.load(name)
        rows.append(
            [
                name,
                kind,
                len(matrix.labels),
                f"{_plain(matrix.lowest)} to {_plain(matrix.highest)}",
                summary,
            ]
        )
    return rows


def _plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
