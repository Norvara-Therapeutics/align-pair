"""Substitution matrices: the built-in tables, user files, and lookup.

A score only means something beside the matrix that produced it. The BLOSUM
tables are in half-bit or third-bit units and the PAM tables in another scale
again, so an alignment scoring 412 under BLOSUM62 and one scoring 412 under
PAM250 are not comparable, and neither is comparable with a count of matches
under a match/mismatch scheme. Every matrix here therefore carries the comment
lines of the file it came from, and every output names the matrix it used.

Two residues can be identical and still score badly. NUC.4.4 gives S against S
minus one, because S stands for C or G and two independent draws from a pair
agree only half the time. That is why identity and similarity are counted
separately everywhere in this package: identity is a property of the letters,
similarity a property of the matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from align_pair import matrix_data
from align_pair.alphabet import GAPS, PROTEIN, STOP, Alphabet

#: The built-in matrices, in the order they are listed.
BUILT_IN: tuple[str, ...] = (
    "BLOSUM45",
    "BLOSUM50",
    "BLOSUM62",
    "BLOSUM80",
    "PAM30",
    "PAM120",
    "PAM250",
    "NUC.4.4",
)

#: The matrix used for protein when none is named. BLOSUM62 is the default of
#: BLAST and of EMBOSS needle, so a score from this package is comparable with
#: one from those when the gap penalties match too.
DEFAULT_PROTEIN: str = "BLOSUM62"

#: The matrix used for nucleic acid when none is named. NUC.4.4 is what EMBOSS
#: ships as EDNAFULL and what needle and water use for DNA.
DEFAULT_NUCLEOTIDE: str = "NUC.4.4"

#: The name under which a match/mismatch scheme is reported.
SIMPLE: str = "simple"


class MatrixError(ValueError):
    """Raised when a matrix cannot be read, or cannot score a sequence."""


@dataclass(frozen=True)
class Matrix:
    """One substitution matrix.

    Attributes:
        name: how the matrix is named in output.
        kind: 'nucleotide' or 'protein'.
        labels: the residues the table is indexed by, in table order.
        scores: the table, as rows in the order of ``labels``.
        comments: the comment lines of the source file, which say at what
            scale the scores are expressed.
        origin: where the matrix came from, for the output to report.
    """

    name: str
    kind: str
    labels: str
    scores: tuple[tuple[float, ...], ...]
    comments: tuple[str, ...] = ()
    origin: str = "built in"

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.labels):
            raise MatrixError(
                f"{self.name}: {len(self.scores)} rows for {len(self.labels)} residues"
            )
        for label, row in zip(self.labels, self.scores):
            if len(row) != len(self.labels):
                raise MatrixError(
                    f"{self.name}: row {label} has {len(row)} scores for "
                    f"{len(self.labels)} residues"
                )
            for residue, value in zip(self.labels, row):
                if not math.isfinite(value):
                    raise MatrixError(
                        f"{self.name}: {label} against {residue} is {value}, and a "
                        "score has to be a finite number"
                    )

    @property
    def index(self) -> dict[str, int]:
        """Residue to its column in the table."""
        return {residue: position for position, residue in enumerate(self.labels)}

    @property
    def is_nucleotide(self) -> bool:
        """Whether this matrix scores nucleic acid."""
        return self.kind == "nucleotide"

    @property
    def symmetric(self) -> bool:
        """Whether the table reads the same both ways round.

        An asymmetric matrix makes the alignment depend on which sequence is
        given first, which is worth saying out loud rather than leaving to be
        discovered from two different scores for one pair.
        """
        size = len(self.labels)
        return all(
            self.scores[row][column] == self.scores[column][row]
            for row in range(size)
            for column in range(row + 1, size)
        )

    @property
    def lowest(self) -> float:
        """The smallest score in the table."""
        return min(min(row) for row in self.scores)

    @property
    def highest(self) -> float:
        """The largest score in the table."""
        return max(max(row) for row in self.scores)

    def covers(self, residue: str) -> bool:
        """Whether the table scores a residue."""
        return residue.upper() in self.index

    def score(self, first: str, second: str) -> float:
        """The score of one residue against another.

        Raises:
            MatrixError: if either residue is not in the table.
        """
        index = self.index
        for residue in (first, second):
            if residue.upper() not in index:
                raise MatrixError(
                    f"{self.name} does not score {residue.upper()!r}"
                )
        return self.scores[index[first.upper()]][index[second.upper()]]

    def row(self, residue: str) -> tuple[float, ...]:
        """One residue's row of the table, indexed by column.

        Raises:
            MatrixError: if the residue is not in the table.
        """
        try:
            return self.scores[self.index[residue.upper()]]
        except KeyError:
            raise MatrixError(f"{self.name} does not score {residue.upper()!r}") from None

    def columns(self, sequence: str) -> list[int]:
        """Each residue of a sequence as its column in the table.

        Raises:
            MatrixError: naming the first residue the table cannot score.
        """
        index = self.index
        positions: list[int] = []
        for offset, residue in enumerate(sequence.upper(), start=1):
            try:
                positions.append(index[residue])
            except KeyError:
                raise MatrixError(
                    f"{self.name} does not score {residue!r}, at position {offset}"
                ) from None
        return positions

    def unscored(self, sequence: str) -> list[str]:
        """The residues of a sequence the table cannot score, sorted."""
        index = self.index
        return sorted({residue for residue in sequence.upper() if residue not in index})


def _table(name: str) -> Matrix:
    return Matrix(
        name=name,
        kind=matrix_data.KINDS[name],
        labels=matrix_data.LABELS[name],
        scores=matrix_data.SCORES[name],
        comments=matrix_data.COMMENTS[name],
        origin="built in",
    )


def simple(match: float, mismatch: float, kind: str) -> Matrix:
    """A match/mismatch matrix over every code of one kind of alphabet.

    Every residue scores ``match`` against itself and ``mismatch`` against
    anything else, ambiguity codes included: under this scheme N against N is a
    match because the letters agree, which is exactly what a match/mismatch
    count means and exactly why the published tables do not do it.
    """
    if kind not in ("nucleotide", "protein"):
        raise MatrixError(f"unknown kind of alphabet: {kind!r}")
    if kind == "nucleotide":
        labels = "ACGTURYSWKMBDHVN"
    else:
        labels = PROTEIN.core + "BZJX" + STOP
    scores = tuple(
        tuple(match if first == second else mismatch for second in labels)
        for first in labels
    )
    comments = (
        f"match/mismatch scheme: {_plain(match)} for identical residues, "
        f"{_plain(mismatch)} for anything else",
        "every ambiguity code is scored as an ordinary letter, so N against N "
        "is a match",
    )
    return Matrix(SIMPLE, kind, labels, scores, comments, origin="match/mismatch")


def _plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def parse(text: str, name: str, origin: str) -> Matrix:
    """Read a matrix in the format the NCBI and EMBOSS publish.

    Comment lines begin with '#', the next line names the columns, and each
    remaining line names its row and gives one score per column.

    Raises:
        MatrixError: if the text is not a readable matrix.
    """
    comments: list[str] = []
    labels = ""
    rows: list[tuple[float, ...]] = []
    order: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line.lstrip("#").strip())
            continue
        fields = line.split()
        if not labels:
            labels = "".join(fields).upper()
            if len(set(labels)) != len(labels):
                repeated = sorted(
                    {residue for residue in labels if labels.count(residue) > 1}
                )
                raise MatrixError(
                    f"{origin}: line {lineno} names {', '.join(repeated)} twice"
                )
            continue
        order.append(fields[0].upper())
        try:
            rows.append(tuple(_value(field) for field in fields[1:]))
        except ValueError as error:
            raise MatrixError(f"{origin}: line {lineno} {error}") from None

    if not labels or not rows:
        raise MatrixError(f"{origin}: no score table found")
    if "".join(order) != labels:
        raise MatrixError(
            f"{origin}: the rows name {''.join(order)} but the columns name {labels}; "
            "a matrix has to be square to score a pair either way round"
        )
    for label, row in zip(order, rows):
        if len(row) != len(labels):
            raise MatrixError(
                f"{origin}: row {label} has {len(row)} scores for "
                f"{len(labels)} columns"
            )

    kind = "nucleotide" if set(labels) <= set("ACGTURYSWKMBDHVN") else "protein"
    return Matrix(name, kind, labels, tuple(rows), tuple(comments), origin)


def _value(field: str) -> float:
    """One score from a matrix file, which has to be a number.

    ``float()`` accepts 'nan' and 'inf', and neither is a score: every
    comparison against NaN is false, so a matrix holding one produces an
    alignment with no columns and a score of NaN, which is not JSON either.
    The arithmetic never reports a problem, so the file has to.
    """
    try:
        return int(field)
    except ValueError:
        pass
    try:
        value = float(field)
    except ValueError:
        raise ValueError(f"holds {field!r}, which is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"holds {field!r}; a score has to be a finite number")
    return value


def read(path: str | Path) -> Matrix:
    """Read a matrix from a file.

    Raises:
        MatrixError: if the file cannot be read or is not a matrix.
    """
    location = Path(path)
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise MatrixError(f"cannot read matrix {path}: {error}") from None
    return parse(text, location.name, origin=str(path))


def load(name: str) -> Matrix:
    """Look a matrix up by built-in name, or read it from a path.

    A built-in name is matched first and case-insensitively, so a file called
    BLOSUM62 in the working directory does not shadow the published table;
    name it './BLOSUM62' to read the file instead.

    Raises:
        MatrixError: if the name is neither a built-in nor a readable file.
    """
    folded = {built.upper(): built for built in BUILT_IN}
    if name.upper() in folded:
        return _table(folded[name.upper()])
    location = Path(name)
    if location.exists():
        return read(location)
    known = ", ".join(BUILT_IN)
    raise MatrixError(
        f"unknown matrix: {name!r}; choose from {known}, or give a path to a "
        "matrix file"
    )


def default_for(alphabet: Alphabet) -> Matrix:
    """The matrix used for an alphabet when none is named."""
    name = DEFAULT_NUCLEOTIDE if alphabet.is_nucleotide else DEFAULT_PROTEIN
    return _table(name)


def fold(sequence: str, matrix: Matrix) -> tuple[str, str]:
    """Rewrite U as T, or T as U, when the matrix carries only one of them.

    NUC.4.4 is indexed by T and has no row for U, so RNA would otherwise be
    unscorable by the only nucleotide matrix published. The substitution is
    exact - U and T pair with A identically - but it is still a substitution,
    so this returns the reason to report alongside the sequence.
    """
    if not matrix.is_nucleotide:
        return sequence, ""
    has_t = matrix.covers("T")
    has_u = matrix.covers("U")
    if has_t and not has_u and "U" in sequence:
        count = sequence.count("U")
        return (
            sequence.replace("U", "T"),
            f"{count} uracil residue{'' if count == 1 else 's'} scored as thymine, "
            f"because {matrix.name} is indexed by T and has no row for U",
        )
    if has_u and not has_t and "T" in sequence:
        count = sequence.count("T")
        return (
            sequence.replace("T", "U"),
            f"{count} thymine residue{'' if count == 1 else 's'} scored as uracil, "
            f"because {matrix.name} is indexed by U and has no row for T",
        )
    return sequence, ""


def describe(matrix: Matrix) -> list[tuple[str, str]]:
    """Label and value pairs describing a matrix, for the output to print."""
    return [
        ("matrix", matrix.name),
        ("scores", "nucleotide" if matrix.is_nucleotide else "protein"),
        ("residues", f"{len(matrix.labels)} ({matrix.labels})"),
        ("range", f"{_plain(matrix.lowest)} to {_plain(matrix.highest)}"),
        ("symmetric", "yes" if matrix.symmetric else "no"),
        ("origin", matrix.origin),
    ]


def grid(matrix: Matrix) -> tuple[list[str], list[list[object]]]:
    """A matrix as a header row and rows of values, for a table."""
    columns = ["", *matrix.labels]
    rows: list[list[object]] = []
    for label, row in zip(matrix.labels, matrix.scores):
        rows.append([label, *row])
    return columns, rows


def unscorable(sequences: Mapping[str, str], matrix: Matrix) -> list[str]:
    """Reasons the matrix cannot score these sequences, empty when it can.

    Named sequences in, one message per sequence out. A gap character is
    reported separately from an unknown residue, because the remedy differs:
    an already-aligned sequence is realigned with --degap, while a residue no
    matrix scores needs a different matrix or a different sequence.
    """
    problems: list[str] = []
    for name, sequence in sequences.items():
        gaps = sorted({residue for residue in sequence if residue in GAPS})
        if gaps:
            shown = ", ".join(repr(character) for character in gaps)
            problems.append(
                f"{name} is already aligned: it carries the gap character(s) "
                f"{shown}; align the residues with --degap, which removes them"
            )
            continue
        missing = matrix.unscored(sequence)
        if missing:
            shown = ", ".join(repr(residue) for residue in missing)
            problems.append(
                f"{matrix.name} does not score {shown}, which {name} carries"
            )
    return problems


def listing() -> list[tuple[str, str, str]]:
    """Every built-in matrix as name, kind and the first comment line."""
    rows: list[tuple[str, str, str]] = []
    for name in BUILT_IN:
        comments: Sequence[str] = matrix_data.COMMENTS[name]
        # Several of the published files open with a bare '#', so the first
        # comment line can be empty; the first one that says something is what
        # a listing wants.
        summary = next((line for line in comments if line), "")
        rows.append((name, matrix_data.KINDS[name], summary))
    return rows
