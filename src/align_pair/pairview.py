"""The side-by-side view of an alignment, in blocks of a fixed width.

Each block carries the two rows with the coordinates they span and a marker
line between them. The markers say which of three different things a column
is, because 'match or not' loses the distinction the matrix exists to make:

    ``|``  the same residue in both rows
    ``:``  different residues the matrix scores above zero
    ``.``  different residues it does not
    (space) a gap

The coordinates are positions in the original records, so a local alignment can
be read against the files it came from. A row that is entirely gaps repeats the
last position before it rather than inventing one.

When the second row is the reverse complement of its record, its coordinates
count DOWN, as they do in a BLAST report: the residues shown are the record's,
read the other way, and a reader checking the region against the file needs
positions in the file rather than positions along a sequence the file does not
hold. Pass ``second_length`` to get that numbering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from align_pair.aligner import Alignment
from align_pair.alphabet import GAP
from align_pair.matrices import Matrix

#: What each marker means, for a legend.
LEGEND: tuple[tuple[str, str], ...] = (
    ("|", "identical residues"),
    (":", "different residues scoring above zero"),
    (".", "different residues scoring zero or below"),
    (" ", "a gap"),
)

#: Marker characters, by what the column is.
IDENTICAL: str = "|"
SIMILAR: str = ":"
DIFFERENT: str = "."
GAPPED: str = " "

#: How many residues a block holds when no width is given.
DEFAULT_WIDTH: int = 60

#: The widest an identifier may be before it is shortened.
LABEL_WIDTH: int = 14


def markers(alignment: Alignment, matrix: Matrix) -> str:
    """The marker line for a whole alignment, one character per column."""
    marks: list[str] = []
    for upper, lower in zip(alignment.aligned_first, alignment.aligned_second):
        if upper == GAP or lower == GAP:
            marks.append(GAPPED)
        elif upper.upper() == lower.upper():
            # Case is soft-masking, not a different residue, and the matrix
            # scores the pair as identical; the marker has to agree with it.
            marks.append(IDENTICAL)
        elif matrix.score(upper, lower) > 0:
            marks.append(SIMILAR)
        else:
            marks.append(DIFFERENT)
    return "".join(marks)


@dataclass(frozen=True)
class Block:
    """One block of the side-by-side view.

    Attributes:
        first: the first sequence's residues in this block, gaps included.
        marks: the marker line.
        second: the second sequence's residues in this block.
        start_first: 1-based position of the first row's first residue, or of
            the last residue before the block when the row is all gaps.
        end_first: 1-based position of its last residue, by the same rule.
        start_second: the same for the second row.
        end_second: the same for the second row.
    """

    first: str
    marks: str
    second: str
    start_first: int
    end_first: int
    start_second: int
    end_second: int


def blocks(
    alignment: Alignment,
    matrix: Matrix,
    width: int = DEFAULT_WIDTH,
    second_length: int | None = None,
) -> list[Block]:
    """Cut an alignment into blocks of at most ``width`` columns.

    Args:
        alignment: the alignment to cut up.
        matrix: the matrix it was scored with, for the marker line.
        width: how many columns a block holds.
        second_length: the length of the second record when the second row
            shows its reverse complement. Given, the second row's coordinates
            are positions in that record and so count down.

    Raises:
        ValueError: if the width is not positive.
    """
    if width < 1:
        raise ValueError(f"block width must be 1 or more, got {width}")
    marks = markers(alignment, matrix)
    found: list[Block] = []
    position_first = max(alignment.start_first - 1, 0)
    position_second = max(alignment.start_second - 1, 0)

    def place(position: int) -> int:
        """A position along the second row, as a position in its record."""
        if second_length is None or not position:
            return position
        return second_length - position + 1

    for offset in range(0, alignment.length, width):
        upper = alignment.aligned_first[offset : offset + width]
        lower = alignment.aligned_second[offset : offset + width]
        residues_first = len(upper) - upper.count(GAP)
        residues_second = len(lower) - lower.count(GAP)
        start_first = position_first + 1 if residues_first else position_first
        start_second = position_second + 1 if residues_second else position_second
        position_first += residues_first
        position_second += residues_second
        found.append(
            Block(
                first=upper,
                marks=marks[offset : offset + width],
                second=lower,
                start_first=start_first,
                end_first=position_first,
                start_second=place(start_second),
                end_second=place(position_second),
            )
        )
    return found


def render(
    alignment: Alignment,
    matrix: Matrix,
    labels: tuple[str, str],
    width: int = DEFAULT_WIDTH,
    indent: str = "  ",
    second_length: int | None = None,
) -> list[str]:
    """The side-by-side view as printable lines.

    Args:
        alignment: the alignment to show.
        matrix: the matrix it was scored with, for the marker line.
        labels: how to name the two rows.
        width: how many columns a block holds.
        indent: what to put at the start of every line.
        second_length: the length of the second record when the second row
            shows its reverse complement, so its coordinates count down.
    """
    if alignment.empty:
        return [f"{indent}no residues are aligned"]
    first_label, second_label = (_shorten(label) for label in labels)
    name_width = max(len(first_label), len(second_label))
    digits = max(
        len(str(alignment.end_first)),
        len(str(second_length or alignment.end_second)),
        1,
    )
    lines: list[str] = []
    for index, block in enumerate(blocks(alignment, matrix, width, second_length)):
        if index:
            lines.append("")
        lines.append(
            f"{indent}{first_label:<{name_width}} {block.start_first:>{digits}} "
            f"{block.first} {block.end_first:>{digits}}"
        )
        lines.append(
            f"{indent}{'':<{name_width}} {'':>{digits}} {block.marks}"
        )
        lines.append(
            f"{indent}{second_label:<{name_width}} {block.start_second:>{digits}} "
            f"{block.second} {block.end_second:>{digits}}"
        )
    return lines


def legend_line() -> str:
    """The marker legend as one line."""
    parts = [
        f"{'space' if mark == ' ' else mark} {meaning}" for mark, meaning in LEGEND
    ]
    return ", ".join(parts)


def fasta(
    alignment: Alignment,
    labels: tuple[str, str],
    descriptions: tuple[str, str] = ("", ""),
    width: int = DEFAULT_WIDTH,
) -> str:
    """The aligned pair as FASTA, gaps included.

    This is the alignment in a form another program can read. Only the aligned
    part is written: residues outside it were not aligned to anything, and
    including them would make the two records disagree in length.
    """
    out: list[str] = []
    rows: Sequence[tuple[str, str, str]] = (
        (labels[0], descriptions[0], alignment.aligned_first),
        (labels[1], descriptions[1], alignment.aligned_second),
    )
    for label, description, sequence in rows:
        header = f">{label} {description}".rstrip()
        out.append(header)
        if not sequence:
            out.append("")
            continue
        for offset in range(0, len(sequence), width):
            out.append(sequence[offset : offset + width])
    return "\n".join(out) + "\n"


def _shorten(label: str) -> str:
    if len(label) <= LABEL_WIDTH:
        return label
    return label[: LABEL_WIDTH - 1] + "\u2026"
