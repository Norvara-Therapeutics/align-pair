"""What an alignment is worth, counted column by column.

Percent identity is the number most often quoted from an alignment and the
number most often quoted without its denominator. Three are in use - identical
columns over the alignment's length, over the shorter sequence, or over the
columns that hold two residues - and they differ by as much as the figure
itself when an alignment is gappy. Two are reported here, each labelled with
what it was divided by, and no bare 'identity' is reported anywhere.

Identity and similarity are counted separately because they are different
claims. Identity is a property of the letters, compared without regard to case
because case is soft-masking and not a different residue: the same residue in
both rows.
Similarity is a property of the matrix: a pair it scores above zero. Neither
contains the other. Leucine against isoleucine is similar and not identical,
and under NUC.4.4 the ambiguity code S against itself is identical and not
similar, because S stands for C or G and two draws from a pair agree half the
time.
"""

from __future__ import annotations

from dataclasses import dataclass

from align_pair.aligner import Alignment
from align_pair.alphabet import GAP, Alphabet
from align_pair.matrices import Matrix


@dataclass(frozen=True)
class Metrics:
    """Counts taken over the columns of one alignment.

    Attributes:
        length: how many columns the alignment has.
        identities: columns whose two residues are the same character.
        similarities: columns the matrix scores above zero, identical or not.
        mismatches: columns holding two residues that differ.
        gaps: columns holding a gap.
        gap_runs: how many separate runs of gaps there are.
        longest_gap: the length of the longest run of gaps.
        gaps_first: gap columns that fall in the first sequence.
        gaps_second: gap columns that fall in the second sequence.
        ambiguous_identities: identical columns whose residue is an ambiguity
            code, so the letters agree but the residues may not.
        identical_unscored: identical columns the matrix does not score above
            zero, which is possible and surprising enough to report.
        length_first: the whole first sequence's length.
        length_second: the whole second sequence's length.
        unaligned_first: residues of the first sequence outside the alignment.
        unaligned_second: residues of the second sequence outside it.
    """

    length: int
    identities: int
    similarities: int
    mismatches: int
    gaps: int
    gap_runs: int
    longest_gap: int
    gaps_first: int
    gaps_second: int
    ambiguous_identities: int
    identical_unscored: int
    length_first: int
    length_second: int
    unaligned_first: int
    unaligned_second: int

    @property
    def shorter(self) -> int:
        """The length of the shorter of the two whole sequences."""
        return min(self.length_first, self.length_second)

    @property
    def identity(self) -> float | None:
        """Identical columns as a share of the alignment's length."""
        return self._share(self.identities, self.length)

    @property
    def identity_of_shorter(self) -> float | None:
        """Identical columns as a share of the shorter whole sequence.

        This is the figure a database search reports, and it answers a
        different question: how much of the smaller molecule is accounted for,
        rather than how well the aligned part agrees.
        """
        return self._share(self.identities, self.shorter)

    @property
    def similarity(self) -> float | None:
        """Columns scoring above zero as a share of the alignment's length."""
        return self._share(self.similarities, self.length)

    @property
    def gap_share(self) -> float | None:
        """Gap columns as a share of the alignment's length."""
        return self._share(self.gaps, self.length)

    @property
    def aligned_columns(self) -> int:
        """Columns holding a residue in both rows."""
        return self.length - self.gaps

    @staticmethod
    def _share(part: int, whole: int) -> float | None:
        """A share, or None when there is nothing to divide by.

        An empty alignment has no identity: reporting 0% would say the residues
        disagree, and reporting 100% that they agree, when neither was looked
        at.
        """
        if whole <= 0:
            return None
        return part / whole


def measure(
    alignment: Alignment,
    matrix: Matrix,
    alphabet: Alphabet,
    length_first: int,
    length_second: int,
) -> Metrics:
    """Count what an alignment contains.

    Args:
        alignment: the alignment to measure.
        matrix: the matrix it was scored with, for judging similarity.
        alphabet: the alphabet, for recognising ambiguity codes.
        length_first: the whole first sequence's length, which may exceed the
            aligned part in every mode but global.
        length_second: the whole second sequence's length.
    """
    identities = 0
    similarities = 0
    mismatches = 0
    gaps = 0
    gap_runs = 0
    longest_gap = 0
    current_gap = 0
    gaps_first = 0
    gaps_second = 0
    ambiguous = 0
    unscored = 0
    aligned_first = 0
    aligned_second = 0

    # Which row the previous column's gap sat in, or an empty string when it
    # held two residues. A gap in one sequence followed by a gap in the other
    # is two gap runs and costs two opening penalties, so tracking only
    # 'the previous column was a gap' would report one run of two and a cost
    # the alignment score does not agree with.
    previous_gap = ""
    for upper, lower in zip(alignment.aligned_first, alignment.aligned_second):
        if upper == GAP or lower == GAP:
            gaps += 1
            if upper == GAP:
                side = "first"
                gaps_first += 1
                aligned_second += 1
            else:
                side = "second"
                gaps_second += 1
                aligned_first += 1
            if side != previous_gap:
                gap_runs += 1
                current_gap = 0
            current_gap += 1
            if current_gap > longest_gap:
                longest_gap = current_gap
            previous_gap = side
            continue
        previous_gap = ""
        aligned_first += 1
        aligned_second += 1
        score = matrix.score(upper, lower)
        if score > 0:
            similarities += 1
        # Case is masking, not identity: a soft-masked repeat is written in
        # lower case and is the same residue. The matrix scores 'a' against 'A'
        # as a match, so counting it as a mismatch would put the identity count
        # at odds with the score the same column contributed to.
        if upper.upper() == lower.upper():
            identities += 1
            if alphabet.is_ambiguous(upper):
                ambiguous += 1
            if score <= 0:
                unscored += 1
        else:
            mismatches += 1

    return Metrics(
        length=alignment.length,
        identities=identities,
        similarities=similarities,
        mismatches=mismatches,
        gaps=gaps,
        gap_runs=gap_runs,
        longest_gap=longest_gap,
        gaps_first=gaps_first,
        gaps_second=gaps_second,
        ambiguous_identities=ambiguous,
        identical_unscored=unscored,
        length_first=length_first,
        length_second=length_second,
        unaligned_first=length_first - aligned_first,
        unaligned_second=length_second - aligned_second,
    )
