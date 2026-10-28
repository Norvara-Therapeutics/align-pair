"""Putting one run together: which sequences, which matrix, what to say about it.

This is where the decisions that change a score are made and recorded. The
alphabet is settled, the matrix chosen or checked against it, RNA folded onto a
matrix indexed by thymine, the strand chosen, and the alignment measured. Every
such decision that a reader needs in order to read the score correctly becomes
a qualification, and the qualifications travel into the terminal summary, the
report and the JSON alike, because a program reading the JSON cannot see a note
printed beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from align_pair import matrices
from align_pair.aligner import (
    MODES,
    Alignment,
    AlignmentError,
    Gaps,
    align,
    cells,
    footprint,
)
from align_pair.alphabet import (
    GAPS,
    Detection,
    degapped,
    disagreements,
    gap_count,
    resolve,
    reverse_complement,
)
from align_pair.matrices import Matrix, MatrixError
from align_pair.metrics import Metrics, measure
from align_pair.reader import Record, SequenceError

#: How two inputs are paired up.
PAIRINGS: tuple[str, ...] = ("all", "positional")

#: Which strand of the second sequence to align.
STRANDS: tuple[str, ...] = ("plus", "minus", "both")

#: How many cells a run may compute before it is refused. At roughly two and a
#: half million cells a second this is about forty seconds of work, which is as
#: much as a command should take without being asked twice.
DEFAULT_MAX_CELLS: int = 100_000_000

#: How many bytes a run may hold before it is refused. Cells measure time and
#: this measures memory, and a band separates the two: it cuts the cells to the
#: width of the band while the dynamic programme's rows still span the whole
#: second sequence, so a run can be quick and still large.
DEFAULT_MAX_MEMORY: int = 1_000_000_000


@dataclass(frozen=True)
class Settings:
    """Everything that decides what an alignment is.

    Attributes:
        mode: one of ``aligner.MODES``.
        gaps: the affine gap penalty.
        matrix: a built-in name or a path, or None for the alphabet's default.
        alphabet: 'auto', or a named alphabet.
        match: the score for identical residues under a match/mismatch scheme.
        mismatch: the score for differing residues under one.
        band: diagonals either side of the main one, global mode only.
        strand: which strand of the second sequence to align.
        degap: remove gap characters from the input before aligning.
        pairing: how two inputs are paired up.
        max_cells: refuse a pair needing more cells than this.
        max_memory: refuse a pair estimated to hold more bytes than this.
    """

    mode: str = "global"
    gaps: Gaps = field(default_factory=Gaps)
    matrix: str | None = None
    alphabet: str = "auto"
    match: float = 5.0
    mismatch: float = -4.0
    band: int | None = None
    strand: str = "plus"
    degap: bool = False
    pairing: str = "all"
    max_cells: int = DEFAULT_MAX_CELLS
    max_memory: int = DEFAULT_MAX_MEMORY

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            known = ", ".join(MODES)
            raise AlignmentError(
                f"unknown alignment mode: {self.mode!r}; choose from {known}"
            )
        if self.strand not in STRANDS:
            known = ", ".join(STRANDS)
            raise AlignmentError(
                f"unknown strand: {self.strand!r}; choose from {known}"
            )
        if self.pairing not in PAIRINGS:
            known = ", ".join(PAIRINGS)
            raise AlignmentError(
                f"unknown pairing: {self.pairing!r}; choose from {known}"
            )


@dataclass(frozen=True)
class Pair:
    """One aligned pair of records.

    Attributes:
        first: the first record, as read.
        second: the second record, as read.
        alignment: the optimal alignment of the prepared sequences.
        metrics: what the alignment contains.
        strand: which strand of the second sequence was aligned.
        other_strand_score: the other strand's score when both were tried.
        span_second: the aligned interval in the second record's own
            coordinates, which differ from the alignment's when the minus
            strand was aligned.
        qualifications: what a reader needs in order to read the score.
    """

    first: Record
    second: Record
    alignment: Alignment
    metrics: Metrics
    strand: str
    other_strand_score: float | None
    span_second: tuple[int, int]
    qualifications: tuple[str, ...]

    @property
    def label(self) -> str:
        """How the pair is named in output."""
        return f"{self.first.identifier} vs {self.second.identifier}"


@dataclass(frozen=True)
class Analysis:
    """A whole run: the settings, the matrix used, and every pair.

    Attributes:
        settings: what was asked for.
        detection: how the alphabet was settled.
        matrix: the matrix every pair was scored with.
        pairs: the aligned pairs, in the order they were formed.
        sources: how the inputs are named in output.
        qualifications: what applies to the run rather than to one pair.
    """

    settings: Settings
    detection: Detection
    matrix: Matrix
    pairs: tuple[Pair, ...]
    sources: tuple[str, ...]
    qualifications: tuple[str, ...]

    @property
    def alphabet_label(self) -> str:
        """How the alphabet is written in output."""
        return self.detection.alphabet.label


def pair_up(
    first: Sequence[Record],
    second: Sequence[Record] | None,
    pairing: str = "all",
) -> list[tuple[Record, Record]]:
    """Decide which records are aligned against which.

    With one input every record is aligned against every later one, which is
    what a file of two records means and what a file of several asks for. With
    two inputs, 'all' crosses them and 'positional' takes them in step.

    Raises:
        SequenceError: if 'positional' is asked for and the counts differ, or
            if a single input holds only one record.
    """
    if second is None:
        if len(first) < 2:
            raise SequenceError(
                "one input with one record has nothing to align it to; give a "
                "second input, or an input holding two or more records"
            )
        return list(combinations(first, 2))
    if pairing == "positional":
        if len(first) != len(second):
            raise SequenceError(
                f"--pairing positional needs the same number of records in each "
                f"input, but one holds {len(first)} and the other {len(second)}"
            )
        return list(zip(first, second))
    return [(one, two) for one in first for two in second]


def choose_matrix(settings: Settings, detection: Detection) -> tuple[Matrix, list[str]]:
    """Settle the matrix, and what has to be said about the choice.

    Raises:
        MatrixError: if the matrix does not score the alphabet in hand.
    """
    notes: list[str] = []
    alphabet = detection.alphabet
    if settings.matrix is None:
        matrix = matrices.default_for(alphabet)
    elif settings.matrix.lower() == matrices.SIMPLE:
        matrix = matrices.simple(settings.match, settings.mismatch, alphabet.kind)
    else:
        matrix = matrices.load(settings.matrix)

    if matrix.kind != alphabet.kind:
        raise MatrixError(
            f"{matrix.name} scores {matrix.kind} residues but the sequences were "
            f"read as {alphabet.label}; name the alphabet with --alphabet if the "
            "detection is wrong, or choose a matrix for it"
        )
    if not matrix.symmetric:
        notes.append(
            f"{matrix.name} is not symmetric, so swapping the two sequences can "
            "change the score"
        )
    return matrix, notes


def prepare(record: Record, settings: Settings, matrix: Matrix) -> tuple[str, list[str]]:
    """The residues to align, in the record's own alphabet.

    Only what the record itself needs is done here: gaps removed if asked, and
    an empty record refused. The substitution onto the matrix's own letters
    happens later, in ``for_scoring``, because it has to come after the strand
    is chosen - a reverse complement taken in the alphabet of the file is
    biology, and folding U onto T is bookkeeping for a particular table, and
    doing them the other way round complements letters the table put there.

    Raises:
        SequenceError: if the record holds no residues.
        MatrixError: if the record is already aligned.
    """
    sequence = record.sequence
    notes: list[str] = []
    if settings.degap:
        removed = gap_count(sequence)
        if removed:
            sequence = degapped(sequence)
            notes.append(
                f"{record.identifier}: {removed} gap character"
                f"{'' if removed == 1 else 's'} removed before aligning, so its "
                "coordinates are positions in the ungapped residues"
            )
    if not sequence:
        raise SequenceError(
            f"{record.identifier} holds no residues to align"
            + (" once its gap characters are removed" if settings.degap else "")
        )
    gaps = sorted({residue for residue in sequence if residue in GAPS})
    if gaps:
        shown = ", ".join(repr(character) for character in gaps)
        raise MatrixError(
            f"{record.identifier} is already aligned: it carries the gap "
            f"character(s) {shown}; align the residues with --degap, which "
            "removes them"
        )
    return sequence, notes


def for_scoring(
    sequence: str, matrix: Matrix, name: str
) -> tuple[str, list[str]]:
    """The residues written in the letters the matrix is indexed by.

    Raises:
        MatrixError: if the matrix cannot score a residue.
    """
    folded, reason = matrices.fold(sequence, matrix)
    notes = [f"{name}: {reason}"] if reason else []
    problems = matrices.unscorable({name: folded}, matrix)
    if problems:
        raise MatrixError(problems[0])
    return folded, notes


def analyse(
    first: Sequence[Record],
    second: Sequence[Record] | None,
    settings: Settings,
    sources: Sequence[str] = (),
) -> Analysis:
    """Align every pair and gather everything the output needs.

    Raises:
        SequenceError: if the records cannot be paired or cannot be aligned.
        MatrixError: if the matrix does not fit the sequences.
        AlignmentError: if a pair is larger than ``settings.max_cells``.
    """
    couples = pair_up(first, second, settings.pairing)
    everything = [record.sequence for record in first] + [
        record.sequence for record in (second or [])
    ]
    detection = resolve(settings.alphabet, *everything)
    named = {
        record.identifier: record.sequence
        for record in list(first) + list(second or [])
    }
    odd = disagreements(named)
    mixed_note = ""
    if odd:
        shown = "; ".join(
            f"{name} looks like {found.alphabet.label} "
            f"({found.nucleotide_share:.0%} nucleotide codes)"
            for name, found in sorted(odd)
        )
        if settings.alphabet == "auto":
            raise SequenceError(
                "the records are not all the same kind of molecule, and one "
                "matrix has to score them all: "
                f"{shown}. Align them separately, or name the alphabet with "
                "--alphabet to score every pair the same way on purpose"
            )
        # Named on the command line, so the run goes ahead - but every pair is
        # scored by one table, and the reader is told which records that table
        # was not chosen for.
        mixed_note = (
            f"the records are not all the same kind of molecule ({shown}), and "
            f"every pair is scored as {detection.alphabet.label} because "
            "--alphabet said so"
        )
    matrix, run_notes = choose_matrix(settings, detection)
    if mixed_note:
        run_notes.append(mixed_note)

    if settings.strand != "plus" and not detection.alphabet.is_nucleotide:
        raise AlignmentError(
            f"--strand {settings.strand} is only meaningful for nucleic acid, and "
            f"these sequences were read as {detection.alphabet.label}"
        )
    if detection.qualification:
        run_notes.append(detection.qualification)
    if settings.alphabet != "auto" and detection.letters:
        expected = detection.alphabet.is_nucleotide
        looks_nucleotide = detection.nucleotide_share >= 0.9
        if expected is not looks_nucleotide:
            run_notes.append(
                f"the sequences were aligned as {detection.alphabet.label} because "
                f"--alphabet said so, although {detection.nucleotide_share:.0%} of "
                "their residues are nucleotide codes"
            )
    if settings.gaps.free:
        run_notes.append(
            "gaps cost nothing under these penalties, so a gap may be inserted "
            "anywhere it does not lose a match and the number of co-optimal "
            "alignments is correspondingly large"
        )
    elif settings.gaps.extend > settings.gaps.open:
        run_notes.append(
            f"each residue of a gap after the first costs {settings.gaps.extend} "
            f"against {settings.gaps.open} for opening one, so long gaps are "
            "penalised more heavily than several short ones"
        )

    pairs = tuple(
        _align_pair(one, two, settings, matrix, detection) for one, two in couples
    )
    return Analysis(
        settings=settings,
        detection=detection,
        matrix=matrix,
        pairs=pairs,
        sources=tuple(sources),
        qualifications=tuple(run_notes),
    )


def _align_pair(
    first: Record,
    second: Record,
    settings: Settings,
    matrix: Matrix,
    detection: Detection,
) -> Pair:
    """Align one pair, on one strand or on whichever scores better."""
    alphabet = detection.alphabet
    upper_residues, notes_first = prepare(first, settings, matrix)
    lower_residues, notes_second = prepare(second, settings, matrix)
    notes = [*notes_first, *notes_second]
    upper, upper_notes = for_scoring(upper_residues, matrix, first.identifier)
    notes.extend(upper_notes)
    lower = lower_residues

    _check_size(
        first.identifier, second.identifier, len(upper), len(lower), settings
    )

    # The complement is taken in the record's own alphabet, before anything is
    # folded onto the matrix's letters. The other order complements a letter
    # the folding put there: RNA folded onto NUC.4.4 becomes T, which the RNA
    # complement table does not translate, while complementing its A yields a U
    # the table cannot score.
    attempts: list[tuple[str, str]] = [("plus", lower_residues)]
    if settings.strand != "plus":
        attempts.append(("minus", reverse_complement(lower_residues, alphabet)))
    if settings.strand == "minus":
        attempts = attempts[1:]

    results: list[tuple[str, Alignment, list[str]]] = []
    for name, residues in attempts:
        scored, fold_notes = for_scoring(residues, matrix, second.identifier)
        results.append(
            (
                name,
                align(
                    upper,
                    scored,
                    matrix,
                    settings.gaps,
                    settings.mode,
                    settings.band,
                ),
                fold_notes,
            )
        )
    # The plus strand is tried first and kept on a tie, so a palindrome and a
    # sequence that is its own reverse complement report the strand a reader
    # would expect rather than whichever comparison happened to run last.
    strand, alignment, chosen_notes = max(results, key=lambda item: item[1].score)
    notes.extend(chosen_notes)
    other = next(
        (value.score for name, value, _notes in results if name != strand), None
    )
    if len(results) > 1:
        if other is not None and other == alignment.score:
            notes.append(
                f"both strands of {second.identifier} were aligned and scored "
                f"the same, {alignment.score:g}; the plus strand is reported, "
                "which is what a sequence equal to its own reverse complement "
                "does rather than a sign that one strand won"
            )
        else:
            notes.append(
                f"both strands of {second.identifier} were aligned; the "
                f"{strand} strand scored better"
                + (
                    f", {alignment.score:g} against {other:g}"
                    if other is not None
                    else ""
                )
            )
    if strand == "minus":
        notes.append(
            f"the second row shows the reverse complement of "
            f"{second.identifier}, so its coordinates are positions in the "
            "record as given and count down"
        )

    span = _span_on_record(alignment, len(lower_residues), strand)
    metrics = measure(
        alignment, matrix, alphabet, len(upper), len(lower_residues)
    )
    notes.extend(_pair_notes(alignment, metrics, matrix, settings))
    return Pair(
        first=first,
        second=second,
        alignment=alignment,
        metrics=metrics,
        strand=strand,
        other_strand_score=other,
        span_second=span,
        qualifications=tuple(notes),
    )


def _check_size(
    first: str, second: str, length_first: int, length_second: int, settings: Settings
) -> None:
    """Refuse a pair too large to run, naming both the work and the memory.

    Raises:
        AlignmentError: if the pair exceeds either limit.
    """
    needed = cells(length_first, length_second, settings.band)
    held = footprint(length_first, length_second, settings.band)
    if needed > settings.max_cells:
        raise AlignmentError(
            f"aligning {first} with {second} needs {needed:,} cells, past the "
            f"{settings.max_cells:,} this run allows; raise the limit with "
            "--max-cells, narrow the work with --band in global mode, or align "
            "shorter sequences"
        )
    if held > settings.max_memory:
        band_note = (
            " A band does not help here: the rows span the second sequence "
            "whatever the band."
            if settings.band is not None
            else ""
        )
        raise AlignmentError(
            f"aligning {first} with {second} would hold about "
            f"{held / 1e6:,.0f} MB, past the {settings.max_memory / 1e6:,.0f} MB "
            f"this run allows; raise the limit with --max-memory, or align "
            f"shorter sequences.{band_note}"
        )


def _span_on_record(alignment: Alignment, length: int, strand: str) -> tuple[int, int]:
    """The aligned interval of the second sequence, in the record's coordinates.

    On the minus strand the alignment's own coordinates count along the reverse
    complement, so they are mapped back: a reader checking the interval against
    the file needs positions in the file.
    """
    start, end = alignment.start_second, alignment.end_second
    if strand == "plus" or not start:
        return start, end
    return length - end + 1, length - start + 1


def _pair_notes(
    alignment: Alignment,
    metrics: Metrics,
    matrix: Matrix,
    settings: Settings,
) -> list[str]:
    """What has to be said about one alignment."""
    notes: list[str] = []
    if alignment.empty:
        notes.append(
            "no pair of subsequences scores above zero, so the best local "
            "alignment is the empty one and its score is zero"
        )
        return notes
    if alignment.co_optimal > 1:
        bound = "at least " if alignment.co_optimal_capped else ""
        notes.append(
            f"{bound}{alignment.co_optimal:,} alignments share this score; the "
            "one shown is reached by preferring an aligned pair to a gap and "
            "continuing a gap to opening another at each step, which makes it "
            "deterministic rather than the best of the set by any further "
            "measure"
        )
    if alignment.band_limited:
        margin = alignment.band_margin
        room = (
            f"; the path left {margin} diagonal{'' if margin == 1 else 's'} of the "
            "band unused on its nearer side"
            if margin is not None
            else ""
        )
        notes.append(
            f"only {alignment.band} diagonal{'' if alignment.band == 1 else 's'} "
            "either side of the main one were filled, so this score is a lower "
            f"bound on the unbanded optimum rather than the optimum{room}"
        )
    if metrics.identical_unscored:
        notes.append(
            f"{metrics.identical_unscored} identical column"
            f"{'' if metrics.identical_unscored == 1 else 's'} "
            f"score zero or below under {matrix.name} and so count towards "
            "identity but not towards similarity"
        )
    if metrics.ambiguous_identities:
        notes.append(
            f"{metrics.ambiguous_identities} identical column"
            f"{'' if metrics.ambiguous_identities == 1 else 's'} "
            "pair an ambiguity code with itself, where the letters agree but the "
            "residues they stand for need not"
        )
    if settings.mode == "local" and metrics.length == 1:
        notes.append(
            "the best local alignment is a single column, which is what a local "
            "alignment of two unrelated sequences looks like"
        )
    return notes
