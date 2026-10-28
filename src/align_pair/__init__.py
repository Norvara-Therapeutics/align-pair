"""Pairwise sequence alignment with affine gaps and published matrices.

Positions are 1-based and inclusive, as they are in a sequence viewer: a region
reported as 101-164 includes both ends and is 64 residues long. A gap of L
residues costs ``open + (L - 1) * extend``, the convention of EMBOSS needle and
water, so a score from this package matches those programs given the same
matrix and penalties; the BLAST family charges one extension more per gap.

Scores are plain numbers in the units of whichever matrix produced them, and
are not comparable across matrices. Identity and similarity are always reported
with the denominator they were divided by, because the three denominators in
common use differ by as much as the figure itself.
"""

# ``aligner`` is the module and ``align`` the function it provides. They are
# deliberately not the same name: binding the function as ``align_pair.align``
# would shadow a module of that name, and every ``import align_pair.align``
# elsewhere would then get the function instead.
from align_pair.aligner import (
    CO_OPTIMAL_CAP,
    MODE_SUMMARY,
    MODES,
    Alignment,
    AlignmentError,
    Gaps,
    align,
    band_diagonals,
    cells,
)
from align_pair.alphabet import (
    ALPHABETS,
    DNA,
    GAP,
    PROTEIN,
    RNA,
    Alphabet,
    Detection,
    degapped,
    detect,
    resolve,
    reverse_complement,
)
from align_pair.analysis import (
    DEFAULT_MAX_CELLS,
    PAIRINGS,
    STRANDS,
    Analysis,
    Pair,
    Settings,
    analyse,
    pair_up,
)
from align_pair.matrices import (
    BUILT_IN,
    DEFAULT_NUCLEOTIDE,
    DEFAULT_PROTEIN,
    SIMPLE,
    Matrix,
    MatrixError,
    load,
    simple,
)
from align_pair.metrics import Metrics, measure
from align_pair.pairview import Block, blocks, markers
from align_pair.reader import Record, SequenceError, read_all, read_records

__version__ = "1.0.0rc3"

__all__ = [
    "ALPHABETS",
    "Alignment",
    "AlignmentError",
    "Alphabet",
    "Analysis",
    "BUILT_IN",
    "Block",
    "CO_OPTIMAL_CAP",
    "DEFAULT_MAX_CELLS",
    "DEFAULT_NUCLEOTIDE",
    "DEFAULT_PROTEIN",
    "DNA",
    "Detection",
    "GAP",
    "Gaps",
    "MODES",
    "MODE_SUMMARY",
    "Matrix",
    "MatrixError",
    "Metrics",
    "PAIRINGS",
    "PROTEIN",
    "Pair",
    "RNA",
    "Record",
    "SIMPLE",
    "STRANDS",
    "SequenceError",
    "Settings",
    "__version__",
    "align",
    "analyse",
    "band_diagonals",
    "blocks",
    "cells",
    "degapped",
    "detect",
    "load",
    "markers",
    "measure",
    "pair_up",
    "read_all",
    "read_records",
    "resolve",
    "reverse_complement",
    "simple",
]
