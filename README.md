# align-pair

Pairwise sequence alignment from the command line: global, glocal, overlap and
local alignment of DNA, RNA or protein, with the published BLOSUM and PAM
matrices, affine gap penalties, and identity and similarity reported with the
denominator they were divided by. Pure Python with no third-party
dependencies, and a self-contained HTML report that opens from a local disk
with no network connection and no JavaScript.

## Install

Install the prerelease from PyPI:

```bash
python -m pip install --pre norvara-align-pair
```

The Python package and command keep the shorter project name:

```python
import align_pair
```

```bash
align-pair --version
```

## Quickstart

```bash
align-pair align -a examples/hbb_human.fasta -b examples/hbb_mouse.fasta
```

```
> NP_000509.1 vs NP_032246.2
  score    638  global  BLOSUM62  gap 10/0.5  protein
  columns  147  identity 118 (80.3%)  similarity 132 (89.8%)  gaps 0 in 0 runs
  aligned  NP_000509.1 1-147 of 147 aa  |  NP_032246.2 1-147 of 147 aa

  NP_000509.1   1 MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK  60
                  |||||..||:||:.||||||.||||||||||||||||||||:|:||||||:..|:|||.|
  NP_032246.2   1 MVHLTDAEKAAVSGLWGKVNADEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNAK  60

  NP_000509.1  61 VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG 120
                  ||||||||:.||:|||.|||:||||||:|||||||||||||||||||||::|.||.||.|
  NP_032246.2  61 VKAHGKKVITAFNDGLNHLDSLKGTFASLSELHCDKLHVDPENFRLLGNMIVIVLGHHLG 120

  NP_000509.1 121 KEFTPPVQAAYQKVVAGVANALAHKYH 147
                  |:|||..|||:||||||||.|||||||
  NP_032246.2 121 KDFTPAAQAAFQKVVAGVAAALAHKYH 147
  | identical residues, : different residues scoring above zero, . different residues scoring zero or below, space a gap
```

Human and mouse beta globin are the same length and close enough that no gap
pays for itself, so the optimal global alignment is gapless and its identity
count can be checked without aligning anything. The marker line distinguishes
identical residues from merely similar ones, which is the distinction the
matrix exists to make.

The same pair of genes as messenger RNA needs a different mode, because the
untranslated ends diverge where the coding sequence does not:

```bash
align-pair align -a examples/hbb_human_mrna.fasta -b examples/hbb_mouse_mrna.fasta -m local --no-view
```

```
> NM_000518.5 vs NM_008220.5
  score    1,967  local  NUC.4.4  gap 10/0.5  DNA
  columns  644  identity 505 (78.4%)  similarity 505 (78.4%)  gaps 32 in 12 runs
  aligned  NM_000518.5 1-628 of 628 bp  |  NM_008220.5 3-630 of 646 bp
  ! 64 alignments share this score; the one shown is reached by preferring an aligned pair to a gap and continuing a gap to opening another at each step, which makes it deterministic rather than the best of the set by any further measure
```

The alphabet is read from the sequences, so the protein pair is scored with
BLOSUM62 and the nucleotide pair with NUC.4.4 without either being named.

## The modes

| Mode | What it aligns |
| --- | --- |
| `global` | both sequences end to end, every gap paid for (Needleman-Wunsch) |
| `glocal` | the first sequence end to end within the second; the second may overhang at either end for free |
| `overlap` | either sequence may overhang at either end for free, so a suffix of one aligns to a prefix of the other |
| `local` | the best-scoring pair of subsequences, beginning and ending with an aligned pair (Smith-Waterman) |

An overhang is not part of the alignment: it is neither scored nor shown, and
the coordinates beside each row say which residues were aligned.

## Options

```
align
-a, --first PATH         the first input, or '-' for standard input
-b, --second PATH        the second input; without it, the records of the
                         first are aligned against each other
    --input-format       auto | fasta | fastq (default: auto)
    --limit N            read only the first N records of each input
    --pairing            all | positional (default: all)
-m, --mode               global | glocal | overlap | local (default: global)
    --matrix NAME        a built-in matrix, 'simple' for match/mismatch, or a
                         path to a matrix file
    --match SCORE        score for identical residues under 'simple'
    --mismatch SCORE     score for differing residues under 'simple'
    --gap-open PENALTY   the first residue of a gap (default: 10)
    --gap-extend PENALTY each further residue of a gap (default: 0.5)
    --alphabet           auto | dna | rna | protein (default: auto)
    --strand             plus | minus | both; nucleic acid only
    --band N             fill only N diagonals either side of the main one,
                         in global mode
    --degap              remove gap characters from the input before aligning
    --max-cells N        refuse a pair needing more cells than this
    --max-memory BYTES   refuse a pair estimated to hold more than this;
                         takes a K, M or G suffix
    --width N            residues per block of the alignment view
    --no-view            keep the metrics, leave out the alignment itself

matrices [NAME]          list the built-in matrices, or print one
```

## Outputs

| Flag | Output |
| --- | --- |
| `--format table` (default) | the readable per-pair summary on stdout |
| `--format tsv` | one row of metrics per pair, on stdout |
| `--format json` \| `none` | every metric and both aligned rows as JSON, or nothing |
| `--output PATH` \| `-` | one row of metrics per pair, as a TSV |
| `--columns PATH` | every column of every alignment, with its score |
| `--alignment PATH` | the aligned pairs as FASTA, gap characters included |
| `--report PATH` | the self-contained HTML report |

Any combination may be requested in a single pass. Compressed input is
recognised from its magic bytes, so gzip, bzip2 and xz files need no flag and
no particular file name.

Every input is opened, and every output destination resolved, checked and
opened, before any input is read, so a run that cannot be written does not
start and a run that fails leaves no file behind: a destination that cannot be
opened - a name too long for the filesystem, a directory with no write
permission - is found before anything is written, and any file the attempt
created is removed again. An output is refused if it names an input under any
name - a hard link and a symlink are the same file, and comparing names alone
would let a run truncate what it is reading - and two outputs are refused if
they name one file, since the second would silently replace the first. A matrix
file counts as an input, because it is read too.

A file written on the minus strand says so in its own header, and the column
table carries a `strand` field, because a data file is read without the
qualifications that were printed beside it.

## Reading the output

- **A gap of L residues costs `open + (L - 1) * extend`.** The first residue
  costs the opening penalty and each further one the extension, which is the
  convention of EMBOSS needle and water; the same matrix and penalties give the
  same score as those programs. BLAST charges one extension more per gap. A gap
  in one sequence and a gap in the other are two runs, each paying to open, and
  the gap counts, the report's running score and the alignment score are the
  same arithmetic.
- **A score means nothing without its matrix.** BLOSUM62 is in half-bit units
  and BLOSUM45 in third-bit units, so 412 under one is not 412 under the other,
  and neither is comparable with a count of matches. Every output names the
  matrix and carries the comment lines of the file it came from.
- **Identity always comes with its denominator**, over the alignment's length
  and over the shorter whole sequence, because the two differ by as much as the
  figure itself when an alignment is gappy. **Identity and similarity are
  different claims:** identity is a property of the letters, similarity of the
  matrix, and neither contains the other. Leucine against isoleucine is similar
  and not identical; under NUC.4.4 the code S against itself is identical and
  not similar, since S stands for C or G and two draws from a pair agree half
  the time. Where an alignment holds such a column the summary says how many.
  **Case is masking, not identity:** `a` against `A` is a match to the matrix
  and to the identity count alike, and the rows keep the case they were given.
- **An optimal alignment is rarely the only one.** The number sharing the
  optimal score is counted and reported whenever it is more than one; past a
  million million it is written as a bound. The one shown is reached by
  preferring an aligned pair to a gap and continuing a gap to opening another
  at each step, which makes it deterministic and tends towards fewer, longer
  gaps - but that is a preference applied step by step, not a second thing
  being optimised, so another co-optimal alignment may have fewer gap runs.
- **A banded score is a lower bound.** Cells outside the band are never
  computed, so the optimum may lie outside it. The output says so whenever the
  band leaves any of the matrix out, and reports how many diagonals of room the
  path had - a diagnostic, not a proof. Widening the band settles it.
- **A local alignment of nothing is reported as nothing.** When no pair of
  subsequences scores above zero the optimum is the empty alignment, scoring
  zero; a pair scoring exactly zero is that same alignment written out.
- **The alphabet is read from composition, not from a letter.** Every IUPAC
  nucleotide code is also an amino-acid code or ambiguity code, so no letter
  settles it; a sequence counts as nucleic acid when at least 90% of its
  letters are A, C, G, T, U or N. The decision is always reported and
  `--alphabet` overrides it. A protein matrix is refused for nucleic acid, and
  so is a batch whose records are not all one kind of molecule: one matrix
  scores every pair, so a single long protein among a file of DNA would
  otherwise carry the DNA pairs into a protein table with nothing failing.
- **A minus-strand row is numbered along its record, descending.** The second
  row shows the reverse complement, and every coordinate beside it - view,
  summary, column table, JSON - is a position in the record as given.
- **Nothing is quietly corrected.** An already-aligned record is refused until
  `--degap`; a residue the matrix cannot score is named with its position; a
  matrix file holding `nan` or `inf` is refused with its line, since neither is
  a score and every comparison against NaN is false; a FASTA header carrying a
  control character is refused rather than printed, since a summary names its
  records and an ANSI escape would be interpreted by the terminal showing it;
  uracil scored against a thymine-indexed matrix is reported as substituted.

Anything a reader needs in order to read a number correctly is printed after a
`!`, shown in the report, and carried in the JSON under `qualifications`,
because a program reading the JSON cannot see a note printed beside it.

## The HTML report

```bash
align-pair align -a examples/aav2_vp1.fasta -b examples/aav9_vp1.fasta --report aav.html -f none
```

One document, no network, no script: the parameters, the metrics, the running
score along the alignment as an SVG, and the alignment itself in blocks with
every column coloured by what it is. The marker line is kept beside the colour
so the document reads in black and white, and every table in it can be
downloaded from the page as the same tab-separated file `--output` would have
written. Output is byte-deterministic for a given input, so two runs of the
same command produce the same bytes.

## Size and speed

Measured on one core: a 1,000 by 1,000 alignment takes 0.35 s and a 3,000 by
3,000 one 3.8 s, at about 2.5 million cells a second.

Memory has two parts, and only one of them a band reduces. The traceback keeps
three bytes per cell, which a band cuts to the width of the band. The dynamic
programme's rows hold a score and a path count for each of three layers, in the
current row and the previous one, and those twelve arrays span the second
sequence whatever the band - about 480 bytes per residue of it, measured across
alignments from 3,000 to 160,000 residues. A 40,000-residue pair at band zero
is therefore 40,001 cells and about 19 MB, which is why `--max-cells` bounds
the work and `--max-memory` bounds the memory, and why a narrow band over a
long pair is quick and still large.

Both limits are applied per pair, as soon as the records have been read and
before any cell is computed. They cannot be applied earlier: how large the work
is depends on the lengths of the records, and a file is usually far longer than
its longest record, so its size says nothing about whether the run is too big.

## Examples

`examples/` holds six real NCBI records, downloaded unmodified: the VP1 capsid
proteins of adeno-associated virus 2 and 9 (`YP_680426.1`, `AAS99264.1`), human
and mouse beta globin (`NP_000509.1`, `NP_032246.2`), and the two globin mRNAs
(`NM_000518.5`, `NM_008220.5`). Re-fetch them with
`python examples/fetch_examples.py`.

## References

- Needleman SB, Wunsch CD (1970). A general method applicable to the search for
  similarities in the amino acid sequence of two proteins. *Journal of
  Molecular Biology* 48:443-453.
- Smith TF, Waterman MS (1981). Identification of common molecular
  subsequences. *Journal of Molecular Biology* 147:195-197.
- Gotoh O (1982). An improved algorithm for matching biological sequences.
  *Journal of Molecular Biology* 162:705-708.
- Henikoff S, Henikoff JG (1992). Amino acid substitution matrices from protein
  blocks. *PNAS* 89:10915-10919.
- Dayhoff MO, Schwartz RM, Orcutt BC (1978). A model of evolutionary change in
  proteins. In *Atlas of Protein Sequence and Structure* 5:345-352.

## License

PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE).
