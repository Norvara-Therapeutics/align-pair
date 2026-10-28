"""The command line.

Every input is opened and every output destination resolved, checked and
opened before any input is read, so a run that cannot be written does not
start and no output is left behind by one that fails. An output is refused if
it names an input under any name - a hard link and a symlink are the same
file, and comparing names alone would let a run truncate what it is reading -
and two outputs are refused if they name one file, since the second would
silently replace the first. A matrix file counts as an input for this purpose,
because it is read too.

How large the work is cannot be settled this early: it depends on the lengths
of the records, which are not known until they have been read. A file is
usually far longer than its longest record, so its size bounds the work from
above and proves nothing about whether the run is too big. That check therefore
happens as soon as the records are in hand and before any cell is computed, not
before the files are opened.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import IO, Any, ContextManager, Iterator, Sequence

from align_pair import matrices, pairview
from align_pair.aligner import CO_OPTIMAL_CAP, Gaps, MODE_SUMMARY, MODES
from align_pair.aligner import AlignmentError
from align_pair.alphabet import CHOICES as ALPHABETS
from align_pair.analysis import (
    DEFAULT_MAX_CELLS,
    DEFAULT_MAX_MEMORY,
    PAIRINGS,
    STRANDS,
    Analysis,
    Pair,
    Settings,
    analyse,
)
from align_pair.matrices import MatrixError
from align_pair.metrics import Metrics
from align_pair.pairview import DEFAULT_WIDTH
from align_pair.reader import FORMATS, SequenceError, read_all, source_name
from align_pair.report import cumulative, render_html
from align_pair.rows import (
    COLUMN_COLUMNS,
    MATRIX_COLUMNS,
    PAIR_COLUMNS,
    all_column_rows,
    matrix_rows,
    pair_rows,
    render,
)
from align_pair.text import plural, position_range, score as score_text, share

#: What may be written to standard output instead of a table.
STDOUT_FORMATS: tuple[str, ...] = ("table", "tsv", "json", "none")


def _non_negative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def _finite(text: str) -> float:
    """A float option that is a number.

    Rejecting NaN here rather than at the point of use matters because every
    comparison against NaN is false: a penalty of NaN would not be caught by
    any bound and would make every score NaN without any step reporting a
    problem.
    """
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text!r}")
    return value


def _penalty(text: str) -> float:
    value = _finite(text)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"gap penalties are given as positive numbers and subtracted, got {value}"
        )
    return value


def _bytes_option(text: str) -> int:
    """A size in bytes, with an optional K, M or G suffix."""
    scales = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    scale = scales.get(text[-1:].upper(), 1)
    digits = text[:-1] if scale != 1 else text
    try:
        value = int(digits)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a whole number of bytes, optionally with a K, M or G "
            f"suffix, got {text!r}"
        ) from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value * scale


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, as a parser."""
    parser = argparse.ArgumentParser(
        prog="align-pair",
        description=(
            "Global, glocal, overlap and local pairwise alignment of DNA, RNA or "
            "protein sequences, with published substitution matrices and affine "
            "gap penalties."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"align-pair {_version()}"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    align_parser = commands.add_parser(
        "align",
        help="align two sequences, or every pair of a set",
        description=(
            "Align sequences pairwise. With one input every record is aligned "
            "against every later one; with two, the records are crossed or taken "
            "in step."
        ),
    )
    align_parser.add_argument(
        "-a",
        "--first",
        metavar="PATH",
        required=True,
        help="the first input, or '-' for standard input",
    )
    align_parser.add_argument(
        "-b",
        "--second",
        metavar="PATH",
        help=(
            "the second input; without it, the records of the first are aligned "
            "against each other"
        ),
    )
    align_parser.add_argument(
        "--input-format",
        choices=FORMATS,
        default="auto",
        help="input format (default: auto)",
    )
    align_parser.add_argument(
        "--limit",
        type=_non_negative,
        default=0,
        metavar="N",
        help="read only the first N records of each input",
    )
    align_parser.add_argument(
        "--pairing",
        choices=PAIRINGS,
        default="all",
        help=(
            "with two inputs, cross every record with every other, or take them "
            "in step (default: all)"
        ),
    )
    align_parser.add_argument(
        "-m",
        "--mode",
        choices=MODES,
        default="global",
        help="which alignment to compute (default: global)",
    )
    align_parser.add_argument(
        "--matrix",
        metavar="NAME",
        help=(
            "a built-in matrix, 'simple' for match/mismatch, or a path to a "
            "matrix file (default: BLOSUM62 for protein, NUC.4.4 for nucleic acid)"
        ),
    )
    align_parser.add_argument(
        "--match",
        type=_finite,
        default=5.0,
        metavar="SCORE",
        help="score for identical residues under --matrix simple (default: 5)",
    )
    align_parser.add_argument(
        "--mismatch",
        type=_finite,
        default=-4.0,
        metavar="SCORE",
        help="score for differing residues under --matrix simple (default: -4)",
    )
    align_parser.add_argument(
        "--gap-open",
        type=_penalty,
        default=10.0,
        metavar="PENALTY",
        help="penalty for the first residue of a gap (default: 10)",
    )
    align_parser.add_argument(
        "--gap-extend",
        type=_penalty,
        default=0.5,
        metavar="PENALTY",
        help="penalty for each further residue of a gap (default: 0.5)",
    )
    align_parser.add_argument(
        "--alphabet",
        choices=ALPHABETS,
        default="auto",
        help="what the sequences are written in (default: auto)",
    )
    align_parser.add_argument(
        "--strand",
        choices=STRANDS,
        default="plus",
        help=(
            "which strand of the second sequence to align; nucleic acid only "
            "(default: plus)"
        ),
    )
    align_parser.add_argument(
        "--band",
        type=_non_negative,
        metavar="N",
        help=(
            "fill only N diagonals either side of the main one; global mode only, "
            "and the score is then a lower bound on the optimum"
        ),
    )
    align_parser.add_argument(
        "--degap",
        action="store_true",
        help="remove gap characters from the input before aligning",
    )
    align_parser.add_argument(
        "--max-cells",
        type=_positive,
        default=DEFAULT_MAX_CELLS,
        metavar="N",
        help=f"refuse a pair needing more cells than this (default: {DEFAULT_MAX_CELLS:,})",
    )
    align_parser.add_argument(
        "--max-memory",
        type=_bytes_option,
        default=DEFAULT_MAX_MEMORY,
        metavar="BYTES",
        help=(
            "refuse a pair estimated to hold more than this; accepts a suffix "
            f"such as 500M or 2G (default: {DEFAULT_MAX_MEMORY // 1_000_000}M)"
        ),
    )
    align_parser.add_argument(
        "--width",
        type=_positive,
        default=DEFAULT_WIDTH,
        metavar="N",
        help=f"residues per block of the alignment view (default: {DEFAULT_WIDTH})",
    )
    align_parser.add_argument(
        "--no-view",
        action="store_true",
        help="leave the alignment itself out of the summary, keeping the metrics",
    )
    _add_output_options(align_parser)

    matrices_parser = commands.add_parser(
        "matrices",
        help="list the built-in matrices, or show one",
        description=(
            "List the substitution matrices this package carries, or print one "
            "with the comment lines of the file it came from."
        ),
    )
    matrices_parser.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="a built-in matrix or a path to a matrix file",
    )
    matrices_parser.add_argument(
        "-f",
        "--format",
        choices=STDOUT_FORMATS,
        default="table",
        help="what to write to standard output (default: table)",
    )
    return parser


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-f",
        "--format",
        choices=STDOUT_FORMATS,
        default="table",
        help="what to write to standard output (default: table)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="one row of metrics per pair, as a TSV; '-' for standard output",
    )
    parser.add_argument(
        "--columns",
        metavar="PATH",
        help="every column of every alignment, as a TSV",
    )
    parser.add_argument(
        "--alignment",
        metavar="PATH",
        help="the aligned pairs as FASTA, gap characters included",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="the self-contained HTML report",
    )


def _version() -> str:
    from align_pair import __version__

    return __version__


def _writer(destination: str, stdout: IO[str]) -> ContextManager[IO[str]]:
    """Open an output path for writing, or hand back standard output for '-'."""
    if destination == "-":
        return contextlib.nullcontext(stdout)
    return open(destination, "w", encoding="utf-8", newline="\n")


@contextlib.contextmanager
def _destinations(
    wanted: Sequence[tuple[str, str | None]], stdout: IO[str]
) -> Iterator[dict[str, IO[str]]]:
    """Open every destination before any of them is written.

    Checking that a path looks writable is not the same as finding out. A name
    too long for the filesystem, a directory with no write permission or a full
    disk is only discovered on open, and discovering it after an earlier output
    has been created leaves a run that failed with a file behind it - half an
    answer, with nothing to say it is half.

    So all of them are opened first, and if any cannot be, the ones this
    function created are closed and removed before the error is raised. A file
    that already existed is left alone: it has not been written to yet, and
    deleting something the run did not create would be worse than failing.
    """
    handles: dict[str, IO[str]] = {}
    opened: list[tuple[IO[str], str, bool]] = []
    try:
        for name, destination in wanted:
            if destination is None:
                continue
            if destination == "-":
                handles[name] = stdout
                continue
            existed = Path(destination).exists()
            handle = open(destination, "w", encoding="utf-8", newline="\n")
            opened.append((handle, destination, existed))
            handles[name] = handle
        yield handles
    except OSError as error:
        for handle, destination, existed in opened:
            with contextlib.suppress(OSError):
                handle.close()
            if not existed:
                with contextlib.suppress(OSError):
                    Path(destination).unlink()
        raise SequenceError(f"cannot write {error}") from error
    finally:
        for handle, _destination, _existed in opened:
            with contextlib.suppress(OSError):
                handle.close()


def _identity(path: Path) -> tuple[int, int] | None:
    """A file's identity on disk, or None when it does not exist yet.

    Two names for one file - a hard link, a symlink, the same path spelled two
    ways - share an inode but not a name, and a check on names alone would let
    an output truncate the input it is reading.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_dev, info.st_ino


def _same_file(one: str, two: str) -> bool:
    """Whether two destinations name one file."""
    if one == "-" or two == "-":
        return one == two
    first, second = Path(one), Path(two)
    first_id, second_id = _identity(first), _identity(second)
    if first_id is not None and second_id is not None:
        return first_id == second_id
    return first.resolve() == second.resolve()


def _check_inputs(sources: Sequence[str]) -> None:
    """Refuse an input that cannot be read, before any is read."""
    if sources.count("-") > 1:
        raise SequenceError("standard input can only be read once")
    for source in sources:
        if source == "-":
            continue
        path = Path(source)
        if path.is_dir():
            raise SequenceError(f"not a file: {source}")
        if not path.exists():
            raise SequenceError(f"no such file: {source}")


def _check_outputs(destinations: Sequence[str | None], inputs: Sequence[str]) -> None:
    """Refuse an output that cannot be written, or that would eat something."""
    named = [destination for destination in destinations if destination]
    to_stdout = [destination for destination in named if destination == "-"]
    if len(to_stdout) > 1:
        raise SequenceError(
            "two outputs cannot share standard output; write one of them to a file"
        )
    for position, destination in enumerate(named):
        if destination == "-":
            continue
        path = Path(destination)
        try:
            if path.is_dir():
                raise SequenceError(f"output is a directory: {destination}")
            if not path.parent.exists():
                raise SequenceError(f"no such directory: {path.parent}")
        except OSError as error:
            # A name the filesystem will not even look at, such as one past
            # its length limit, fails on stat before it fails on open.
            raise SequenceError(f"cannot write {destination}: {error}") from None
        for earlier in named[:position]:
            if earlier != "-" and _same_file(earlier, destination):
                also = f" and {destination}" if earlier != destination else " twice"
                raise SequenceError(
                    f"two outputs name the same file: {earlier}{also}; the second "
                    "would replace the first"
                )
        for source in inputs:
            if source == "-":
                continue
            if Path(source).exists() and _same_file(destination, source):
                raise SequenceError(
                    f"output {destination} is the input {source}; writing it would "
                    "truncate what is being read"
                )


def _file_outputs(arguments: argparse.Namespace) -> list[str | None]:
    return [
        arguments.output,
        arguments.columns,
        arguments.alignment,
        arguments.report,
    ]


def _settings(arguments: argparse.Namespace) -> Settings:
    return Settings(
        mode=arguments.mode,
        gaps=Gaps(arguments.gap_open, arguments.gap_extend),
        matrix=arguments.matrix,
        alphabet=arguments.alphabet,
        match=arguments.match,
        mismatch=arguments.mismatch,
        band=arguments.band,
        strand=arguments.strand,
        degap=arguments.degap,
        pairing=arguments.pairing,
        max_cells=arguments.max_cells,
        max_memory=arguments.max_memory,
    )


def _ineffective(arguments: argparse.Namespace, stderr: IO[str]) -> None:
    """Report an option that was set but cannot apply.

    No option is silently ignored: an option that cannot be honoured is named
    with the reason, because a user who asked for a band and got a full matrix
    has a wrong idea of what was computed.
    """
    if arguments.matrix is not None and str(arguments.matrix).lower() != matrices.SIMPLE:
        for name, value, default in (
            ("--match", arguments.match, 5.0),
            ("--mismatch", arguments.mismatch, -4.0),
        ):
            if value != default:
                print(
                    f"align-pair: {name} has no effect, because it applies only to "
                    f"--matrix simple and {arguments.matrix} was asked for",
                    file=stderr,
                )
    if arguments.no_view and arguments.format != "table":
        print(
            "align-pair: --no-view has no effect, because the alignment is only "
            f"shown in the table summary and --format {arguments.format} was asked "
            "for",
            file=stderr,
        )
    if arguments.width != DEFAULT_WIDTH and (
        arguments.no_view or arguments.format != "table"
    ) and not arguments.alignment:
        print(
            "align-pair: --width has no effect, because nothing that wraps "
            "residues is being written",
            file=stderr,
        )


def _summary(
    analysis: Analysis,
    stream: IO[str],
    width: int,
    view: bool,
) -> None:
    """The readable per-pair summary."""
    settings = analysis.settings
    matrix = analysis.matrix
    alphabet = analysis.detection.alphabet
    for note in analysis.qualifications:
        print(f"! {note}", file=stream)
    for pair in analysis.pairs:
        metrics = pair.metrics
        alignment = pair.alignment
        start_second, end_second = pair.span_second
        print(
            f"> {pair.first.identifier} vs {pair.second.identifier}",
            file=stream,
        )
        print(
            f"  score    {score_text(alignment.score)}"
            f"  {settings.mode}  {matrix.name}"
            f"  gap {settings.gaps.open:g}/{settings.gaps.extend:g}"
            f"  {alphabet.label}",
            file=stream,
        )
        print(
            f"  columns  {metrics.length:,}"
            f"  identity {metrics.identities:,} ({share(metrics.identity)})"
            f"  similarity {metrics.similarities:,} ({share(metrics.similarity)})"
            f"  gaps {metrics.gaps:,} in {plural(metrics.gap_runs, 'run')}",
            file=stream,
        )
        strand = f" ({pair.strand} strand)" if pair.strand != "plus" else ""
        print(
            f"  aligned  {pair.first.identifier} "
            f"{position_range(alignment.start_first, alignment.end_first)} of "
            f"{metrics.length_first:,} {alphabet.unit}"
            f"  |  {pair.second.identifier} "
            f"{position_range(start_second, end_second)} of "
            f"{metrics.length_second:,} {alphabet.unit}{strand}",
            file=stream,
        )
        if view and not alignment.empty:
            print(file=stream)
            for line in pairview.render(
                alignment,
                matrix,
                (pair.first.identifier, pair.second.identifier),
                width,
                second_length=(
                    metrics.length_second if pair.strand == "minus" else None
                ),
            ):
                print(line, file=stream)
            print(f"  {pairview.legend_line()}", file=stream)
        for note in pair.qualifications:
            print(f"  ! {note}", file=stream)


def _metrics_payload(metrics: Metrics) -> dict[str, Any]:
    return {
        "length": metrics.length,
        "identities": metrics.identities,
        "identity_of_alignment": metrics.identity,
        "identity_of_shorter": metrics.identity_of_shorter,
        "similarities": metrics.similarities,
        "similarity_of_alignment": metrics.similarity,
        "mismatches": metrics.mismatches,
        "gaps": metrics.gaps,
        "gap_of_alignment": metrics.gap_share,
        "gap_runs": metrics.gap_runs,
        "longest_gap": metrics.longest_gap,
        "gaps_in_first": metrics.gaps_first,
        "gaps_in_second": metrics.gaps_second,
        "aligned_columns": metrics.aligned_columns,
        "ambiguous_identities": metrics.ambiguous_identities,
        "identical_unscored": metrics.identical_unscored,
        "unaligned_first": metrics.unaligned_first,
        "unaligned_second": metrics.unaligned_second,
    }


def _pair_payload(pair: Pair, analysis: Analysis) -> dict[str, Any]:
    alignment = pair.alignment
    start_second, end_second = pair.span_second
    return {
        "first": {
            "identifier": pair.first.identifier,
            "description": pair.first.description,
            "length": pair.metrics.length_first,
            "start": alignment.start_first or None,
            "end": alignment.end_first or None,
        },
        "second": {
            "identifier": pair.second.identifier,
            "description": pair.second.description,
            "length": pair.metrics.length_second,
            "start": start_second or None,
            "end": end_second or None,
            "strand": pair.strand,
            "other_strand_score": pair.other_strand_score,
        },
        "score": alignment.score,
        "metrics": _metrics_payload(pair.metrics),
        "alignment": {
            "first": alignment.aligned_first,
            "second": alignment.aligned_second,
            "markers": pairview.markers(alignment, analysis.matrix),
        },
        "co_optimal": alignment.co_optimal,
        "co_optimal_at_least": alignment.co_optimal_capped,
        "cells": alignment.cells,
        "band": alignment.band,
        "band_limited": alignment.band_limited,
        "band_margin": alignment.band_margin,
        "running_score": [
            value for _column, value in cumulative(pair, analysis.matrix, analysis.settings.gaps)
        ],
        "qualifications": list(pair.qualifications),
    }


def _json_payload(analysis: Analysis) -> dict[str, Any]:
    settings = analysis.settings
    return {
        "align_pair": _version(),
        "mode": settings.mode,
        "mode_means": MODE_SUMMARY[settings.mode],
        "alphabet": analysis.detection.alphabet.name,
        "alphabet_named": analysis.detection.requested,
        "nucleotide_share": analysis.detection.nucleotide_share,
        "matrix": {
            "name": analysis.matrix.name,
            "kind": analysis.matrix.kind,
            "origin": analysis.matrix.origin,
            "residues": analysis.matrix.labels,
            "symmetric": analysis.matrix.symmetric,
            "comments": list(analysis.matrix.comments),
        },
        "gap_open": settings.gaps.open,
        "gap_extend": settings.gaps.extend,
        "gap_model": "open + (length - 1) * extend",
        "co_optimal_cap": CO_OPTIMAL_CAP,
        "sources": list(analysis.sources),
        "qualifications": list(analysis.qualifications),
        "pairs": [_pair_payload(pair, analysis) for pair in analysis.pairs],
    }


def _stdout_shared(arguments: argparse.Namespace) -> bool:
    return any(destination == "-" for destination in _file_outputs(arguments))


def run_align(arguments: argparse.Namespace, stdout: IO[str], stderr: IO[str]) -> int:
    """Run the ``align`` command."""
    sources = [arguments.first]
    if arguments.second:
        sources.append(arguments.second)
    reads = list(sources)
    if arguments.matrix and Path(arguments.matrix).exists():
        # The matrix file is read, so an output aimed at it would destroy it
        # before it could be read.
        reads.append(arguments.matrix)
    _check_inputs(sources)
    _check_outputs(_file_outputs(arguments), reads)
    if _stdout_shared(arguments) and arguments.format != "none":
        raise SequenceError(
            "an output is going to standard output, so the summary has to be "
            "silenced with --format none"
        )
    _ineffective(arguments, stderr)

    wanted = [
        ("output", arguments.output),
        ("columns", arguments.columns),
        ("alignment", arguments.alignment),
        ("report", arguments.report),
    ]
    with _destinations(wanted, stdout) as handles:
        first = read_all(arguments.first, arguments.input_format, arguments.limit)
        second = (
            read_all(arguments.second, arguments.input_format, arguments.limit)
            if arguments.second
            else None
        )
        names = [source_name(source) for source in sources]
        analysis = analyse(first, second, _settings(arguments), names)

        if "output" in handles:
            handles["output"].write(render(PAIR_COLUMNS, pair_rows(analysis)))
        if "columns" in handles:
            handles["columns"].write(
                render(COLUMN_COLUMNS, all_column_rows(analysis))
            )
        if "alignment" in handles:
            for pair in analysis.pairs:
                handles["alignment"].write(
                    pairview.fasta(
                        pair.alignment,
                        (pair.first.identifier, pair.second.identifier),
                        _alignment_descriptions(pair),
                        arguments.width,
                    )
                )
        if "report" in handles:
            handles["report"].write(render_html(analysis, _version()))

        if arguments.format == "table":
            _summary(analysis, stdout, arguments.width, not arguments.no_view)
        elif arguments.format == "tsv":
            stdout.write(render(PAIR_COLUMNS, pair_rows(analysis)))
        elif arguments.format == "json":
            json.dump(_json_payload(analysis), stdout, indent=2, sort_keys=False)
            stdout.write("\n")
    return 0


def _alignment_descriptions(pair: Pair) -> tuple[str, str]:
    """The two descriptions for the aligned FASTA, marking a flipped row.

    A file of aligned residues has no qualifications beside it, so a reverse
    complement has to say so in the only place a reader will look: its own
    header. Without it the second record is the record's identifier against
    residues the record does not contain.
    """
    second = pair.second.description
    if pair.strand == "minus":
        marked = "reverse complement" + (f"; {second}" if second else "")
        return pair.first.description, marked
    return pair.first.description, second


def run_matrices(arguments: argparse.Namespace, stdout: IO[str]) -> int:
    """Run the ``matrices`` command."""
    if arguments.name:
        matrix = matrices.load(arguments.name)
        if arguments.format == "json":
            json.dump(
                {
                    "name": matrix.name,
                    "kind": matrix.kind,
                    "origin": matrix.origin,
                    "residues": matrix.labels,
                    "symmetric": matrix.symmetric,
                    "lowest": matrix.lowest,
                    "highest": matrix.highest,
                    "comments": list(matrix.comments),
                    "scores": {
                        residue: dict(zip(matrix.labels, row))
                        for residue, row in zip(matrix.labels, matrix.scores)
                    },
                },
                stdout,
                indent=2,
            )
            stdout.write("\n")
            return 0
        columns, rows = matrices.grid(matrix)
        if arguments.format == "tsv":
            stdout.write(render(columns, rows))
            return 0
        if arguments.format == "none":
            return 0
        print(f"> {matrix.name}", file=stdout)
        for label, value in matrices.describe(matrix):
            print(f"  {label:<10} {value}", file=stdout)
        for comment in matrix.comments:
            if comment:
                print(f"  ! {comment}", file=stdout)
        print(file=stdout)
        widest = max(len(str(value)) for row in rows for value in row[1:])
        header = "     " + "".join(f"{label:>{widest + 1}}" for label in matrix.labels)
        print(header, file=stdout)
        for label, row in zip(matrix.labels, matrix.scores):
            body = "".join(f"{value:>{widest + 1}}" for value in row)
            print(f"  {label}  {body}", file=stdout)
        return 0

    listing = matrices.listing()
    if arguments.format == "json":
        json.dump(
            [
                {"name": name, "kind": kind, "summary": summary}
                for name, kind, summary in listing
            ],
            stdout,
            indent=2,
        )
        stdout.write("\n")
        return 0
    if arguments.format == "tsv":
        stdout.write(render(MATRIX_COLUMNS, matrix_rows(listing)))
        return 0
    if arguments.format == "none":
        return 0
    print(f"> {len(listing)} built-in matrices", file=stdout)
    for name, kind, summary in listing:
        print(f"  {name:<10} {kind:<11} {summary}", file=stdout)
    print(
        "  the default is BLOSUM62 for protein and NUC.4.4 for nucleic acid; "
        "'simple' scores matches and mismatches",
        file=stdout,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line, returning the process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "align":
            return run_align(arguments, sys.stdout, sys.stderr)
        return run_matrices(arguments, sys.stdout)
    except (SequenceError, MatrixError, AlignmentError) as error:
        print(f"align-pair: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"align-pair: {error}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Downstream closed the pipe, as 'head' does; that is not an error.
        with contextlib.suppress(OSError):
            sys.stdout.close()
        return 0
    except OSError as error:
        # A name too long for the filesystem, a full disk, a directory that
        # cannot be read: the operating system's refusals are this program's
        # exit 2, not a traceback.
        print(f"align-pair: {error}", file=sys.stderr)
        return 2
