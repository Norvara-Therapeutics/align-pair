"""Residue alphabets, and deciding which one a pair of sequences is written in.

The alphabet decides which substitution matrix is used, and so decides every
score the package reports. It is detected from composition rather than guessed
from a file name, because no letter settles it: every IUPAC nucleotide code is
also an amino-acid code or an amino-acid ambiguity code, so there is no letter
whose presence proves a sequence is DNA.

Detection is therefore by share - a sequence counts as nucleotide when almost
all of its letters are A, C, G, T, U or N - and the decision is always
reported, never silent. The two mistakes do not cost the same: scoring DNA with
a protein matrix produces a plausible-looking number from a matrix that knows
nothing about transitions and transversions, which is worse than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: Alphabet names accepted on the command line, with 'auto' for detection.
CHOICES: tuple[str, ...] = ("auto", "dna", "rna", "protein")

#: Characters that stand for a missing or deleted residue in an aligned FASTA.
GAPS: frozenset[str] = frozenset("-.~")

#: The gap character this package writes into an alignment.
GAP: str = "-"

#: The translation stop written inside a protein sequence. Not a residue.
STOP: str = "*"

_DNA_AMBIGUITY: dict[str, str] = {
    "R": "AG",
    "Y": "CT",
    "S": "CG",
    "W": "AT",
    "K": "GT",
    "M": "AC",
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "V": "ACG",
    "N": "ACGT",
}

_RNA_AMBIGUITY: dict[str, str] = {
    code: residues.replace("T", "U") for code, residues in _DNA_AMBIGUITY.items()
}

#: Protein ambiguity codes. B, Z and J each stand for a pair that Edman
#: degradation or mass spectrometry cannot separate; X stands for any residue.
_PROTEIN_AMBIGUITY: dict[str, str] = {
    "B": "DN",
    "Z": "EQ",
    "J": "IL",
    "X": "ACDEFGHIKLMNPQRSTVWY",
}

#: The two genetically encoded residues beyond the twenty. They are valid
#: sequence and are recognised as such, but no published substitution matrix
#: scores them, so an alignment containing one has to say so.
_PROTEIN_RARE: dict[str, str] = {
    "U": "selenocysteine",
    "O": "pyrrolysine",
}


@dataclass(frozen=True)
class Alphabet:
    """One residue alphabet.

    Attributes:
        name: the short name used on the command line.
        label: how the alphabet is written in output.
        kind: 'nucleotide' or 'protein'. A matrix is built for one or the
            other and the two are not interchangeable.
        core: the unambiguous residues, in conventional order.
        ambiguity: code to the residues it stands for.
        rare: residues recognised as valid but carried by no matrix.
    """

    name: str
    label: str
    kind: str
    core: str
    ambiguity: Mapping[str, str]
    rare: Mapping[str, str]

    @property
    def is_nucleotide(self) -> bool:
        """Whether this alphabet describes nucleic acid."""
        return self.kind == "nucleotide"

    @property
    def unit(self) -> str:
        """How a length in this alphabet is written."""
        return "bp" if self.is_nucleotide else "aa"

    @property
    def valid(self) -> frozenset[str]:
        """Every character this alphabet accepts.

        A nucleotide alphabet does not accept the stop character: '*' in a
        nucleotide file is not a nucleotide.
        """
        extra = {STOP} if not self.is_nucleotide else set()
        return frozenset(
            set(self.core) | set(self.ambiguity) | set(self.rare) | set(GAPS) | extra
        )

    def is_ambiguous(self, residue: str) -> bool:
        """Whether a residue stands for more than one residue."""
        return residue.upper() in self.ambiguity


DNA = Alphabet(
    name="dna",
    label="DNA",
    kind="nucleotide",
    core="ACGT",
    ambiguity=_DNA_AMBIGUITY,
    rare={},
)

RNA = Alphabet(
    name="rna",
    label="RNA",
    kind="nucleotide",
    core="ACGU",
    ambiguity=_RNA_AMBIGUITY,
    rare={},
)

PROTEIN = Alphabet(
    name="protein",
    label="protein",
    kind="protein",
    core="ACDEFGHIKLMNPQRSTVWY",
    ambiguity=_PROTEIN_AMBIGUITY,
    rare=_PROTEIN_RARE,
)

#: Every alphabet, by name.
ALPHABETS: dict[str, Alphabet] = {
    alphabet.name: alphabet for alphabet in (DNA, RNA, PROTEIN)
}

#: Characters any alphabet might accept. The reader checks against this,
#: because the alphabet is not known until the sequences have been read.
ANY_RESIDUE: frozenset[str] = frozenset(
    set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set(GAPS) | {STOP}
)

#: Residues that count as evidence of a nucleotide sequence.
_NUCLEOTIDE_EVIDENCE: frozenset[str] = frozenset("ACGTUN")

#: The share of letters that must be nucleotide codes for a sequence to be
#: read as nucleotide. Protein rich in alanine, cysteine, glycine and threonine
#: reaches a surprising share, so the bar is set high.
NUCLEOTIDE_THRESHOLD: float = 0.9

#: Below this many letters, composition carries little information and the
#: detected alphabet is reported as weakly supported rather than as a fact.
SHORT_SEQUENCE: int = 20

_COMPLEMENT = {
    "dna": str.maketrans(
        "ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn"
    ),
    "rna": str.maketrans(
        "ACGURYSWKMBDHVNacguryswkmbdhvn", "UGCAYRSWMKVHDBNugcayrswmkvhdbn"
    ),
}


@dataclass(frozen=True)
class Detection:
    """The outcome of deciding what a pair of sequences is written in.

    Attributes:
        alphabet: the alphabet chosen.
        nucleotide_share: the share of letters that are nucleotide codes.
        letters: how many letters the decision was taken over.
        certain: False when the evidence is thin enough that the caller should
            say so rather than report the alphabet plainly.
        requested: True when the caller named the alphabet and nothing was
            detected at all.
    """

    alphabet: Alphabet
    nucleotide_share: float
    letters: int
    certain: bool
    requested: bool = False

    @property
    def qualification(self) -> str:
        """Why the detection is uncertain, or an empty string when it is not."""
        if self.certain:
            return ""
        if not self.letters:
            return "there are no residues to judge from, so DNA was assumed"
        return (
            f"the alphabet was read from only {self.letters} residues, "
            f"{self.nucleotide_share:.0%} of them nucleotide codes; name it with "
            "--alphabet if that is wrong"
        )


def get(name: str) -> Alphabet:
    """Look an alphabet up by name.

    Raises:
        KeyError: if the name is not one of the known alphabets.
    """
    try:
        return ALPHABETS[name.lower()]
    except KeyError:
        known = ", ".join(sorted(ALPHABETS))
        raise KeyError(f"unknown alphabet: {name!r}; choose from {known}") from None


def census(*sequences: str) -> dict[str, int]:
    """Count the residues of one or more sequences, case-insensitively."""
    counts: dict[str, int] = {}
    for sequence in sequences:
        for residue in sequence.upper():
            counts[residue] = counts.get(residue, 0) + 1
    return counts


def detect(counts: Mapping[str, int]) -> Detection:
    """Decide which alphabet a residue census is written in.

    Nucleotide is chosen when at least ``NUCLEOTIDE_THRESHOLD`` of the letters
    are nucleotide codes; RNA rather than DNA when the sequence carries more U
    than T. A sequence with both in quantity is read as DNA, since that is what
    a file mixing the two conventions almost always is.
    """
    letters: dict[str, int] = {}
    for residue, count in counts.items():
        upper = residue.upper()
        if upper in GAPS or upper == STOP:
            continue
        letters[upper] = letters.get(upper, 0) + count

    total = sum(letters.values())
    if not total:
        return Detection(DNA, 0.0, 0, certain=False)

    nucleotide = sum(
        count for residue, count in letters.items() if residue in _NUCLEOTIDE_EVIDENCE
    )
    share = nucleotide / total
    if share < NUCLEOTIDE_THRESHOLD:
        return Detection(PROTEIN, share, total, certain=total >= SHORT_SEQUENCE)

    thymine = letters.get("T", 0)
    uracil = letters.get("U", 0)
    alphabet = RNA if uracil > thymine else DNA
    return Detection(alphabet, share, total, certain=total >= SHORT_SEQUENCE)


def resolve(requested: str, *sequences: str) -> Detection:
    """Settle the alphabet, by request or by detection.

    A named alphabet is taken as given and the share is still measured, so the
    output can report that a sequence named as protein is 97% nucleotide codes
    rather than leaving the reader to notice.

    Raises:
        KeyError: if ``requested`` is not 'auto' or a known alphabet.
    """
    counts = census(*sequences)
    detected = detect(counts)
    if requested == "auto":
        return detected
    return Detection(
        get(requested),
        detected.nucleotide_share,
        detected.letters,
        certain=True,
        requested=True,
    )


def disagreements(sequences: Mapping[str, str]) -> list[tuple[str, Detection]]:
    """Records whose own composition disagrees with the rest on what they are.

    A run scores every pair with one matrix, so the whole batch has to be one
    kind of molecule. Detecting over the records together would let one long
    protein carry a file of DNA into a protein matrix, where every pair -
    including the DNA against DNA - is then scored by a table that knows
    nothing about transitions and transversions, and nothing would fail.

    Returns the records of the minority kind, each with its own detection, or
    an empty list when they agree. Named sequences in, so the caller can say
    which records it means.
    """
    detected = {
        name: detect(census(sequence)) for name, sequence in sequences.items()
    }
    kinds = {found.alphabet.kind for found in detected.values()}
    if len(kinds) < 2:
        return []
    counts = {
        kind: sum(1 for found in detected.values() if found.alphabet.kind == kind)
        for kind in kinds
    }
    minority = min(counts, key=lambda kind: (counts[kind], kind))
    return [
        (name, found)
        for name, found in detected.items()
        if found.alphabet.kind == minority
    ]


def unknown_residues(counts: Mapping[str, int], alphabet: Alphabet) -> list[str]:
    """Residues in a census that the alphabet does not accept, sorted."""
    accepted = alphabet.valid
    return sorted(
        {residue.upper() for residue in counts if residue.upper() not in accepted}
    )


def reverse_complement(sequence: str, alphabet: Alphabet) -> str:
    """Reverse-complement a nucleotide sequence, ambiguity codes included.

    Gaps and any character outside the alphabet are passed through unchanged,
    so the result has the same length as the input.

    Raises:
        ValueError: if the alphabet has no complement.
    """
    if not alphabet.is_nucleotide:
        raise ValueError(f"{alphabet.label} has no complement")
    return sequence.translate(_COMPLEMENT[alphabet.name])[::-1]


def degapped(sequence: str) -> str:
    """The sequence with every gap character removed."""
    return "".join(residue for residue in sequence if residue not in GAPS)


def gap_count(sequence: str) -> int:
    """How many gap characters a sequence carries."""
    return sum(1 for residue in sequence if residue in GAPS)
