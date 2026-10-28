"""FASTA and FASTQ input.

An alignment needs both sequences whole, so records are held in memory rather
than streamed: the dynamic programme is indexed by position in each sequence
and cannot be computed a line at a time. That is also why this package reads
single records and small files rather than whole genomes.

Plain text, gzip, bzip2 and xz are accepted from a path or from standard
input, and the compression is recognised from the file's magic bytes rather
than its name. Characters are checked against the union of the alphabets, so a
file that is not sequence at all is refused rather than aligned; the stricter
check against a particular alphabet happens once the alphabet is known.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterable, Iterator

from align_pair.alphabet import ANY_RESIDUE


class SequenceError(ValueError):
    """Raised when input cannot be parsed, or cannot be aligned as asked."""


#: The input formats this reader understands, plus sniffing.
FORMATS: tuple[str, ...] = ("auto", "fasta", "fastq")

#: Every character a sequence line may carry, in either case.
VALID_RESIDUES: frozenset[str] = frozenset(
    {character.upper() for character in ANY_RESIDUE}
    | {character.lower() for character in ANY_RESIDUE}
)

#: Deleting every valid residue leaves only the offending characters, which is
#: a C-speed check and so affordable on every line of a large file.
_RESIDUE_SIEVE = str.maketrans("", "", "".join(sorted(VALID_RESIDUES)))

#: Quality strings are printable ASCII; every Phred encoding sits inside it.
_QUALITY_SIEVE = str.maketrans("", "", "".join(chr(code) for code in range(33, 127)))

#: Compression recognised from the first bytes of the stream, with the opener
#: that wraps an already-open binary stream rather than a file name.
_MAGIC: tuple[tuple[bytes, Callable[[IO[bytes]], IO[bytes]]], ...] = (
    (b"\x1f\x8b", lambda stream: gzip.GzipFile(fileobj=stream)),
    (b"BZh", bz2.BZ2File),
    (b"\xfd7zXZ\x00", lzma.LZMAFile),
)


def _reject(line: str, lineno: int, sieve: dict[int, int | None], what: str) -> None:
    invalid = sorted(set(line.translate(sieve)))
    if not invalid:
        return
    shown = ", ".join(repr(character) for character in invalid[:5])
    if len(invalid) > 5:
        shown += ", ..."
    raise SequenceError(f"line {lineno}: {what}; unexpected character(s): {shown}")


def _check_residues(line: str, lineno: int) -> None:
    """Reject sequence data that is not residue text.

    Without this, a file that is not a sequence at all - a column of numbers,
    a saved error page - is absorbed as residues and aligned, which is worse
    than an error. Prose written only in letters cannot be caught this way,
    since every letter is an amino-acid code.
    """
    _reject(line, lineno, _RESIDUE_SIEVE, "not sequence data")


def _check_quality(line: str, lineno: int) -> None:
    _reject(line, lineno, _QUALITY_SIEVE, "not a quality string")


@dataclass(frozen=True)
class Record:
    """One FASTA or FASTQ record.

    Attributes:
        identifier: text up to the first whitespace of the header line.
        description: remainder of the header line, empty when absent.
        sequence: residues, uppercased, with all whitespace removed.
        quality: the quality string of a FASTQ record, None for FASTA.
    """

    identifier: str
    description: str
    sequence: str
    quality: str | None = None

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def header(self) -> str:
        """The header line without its '>' or '@' prefix."""
        return f"{self.identifier} {self.description}".strip()


#: Characters a header line may not carry. A FASTA header is a line of
#: printable text, and a control character in one is either a damaged file or a
#: terminal escape sequence aimed at whoever prints the identifier - an
#: alignment summary names its records, so a header holding an ANSI erase or an
#: OSC title-set would run in the reader's terminal.
_CONTROL = frozenset(chr(code) for code in range(32)) | {chr(127)}


def _check_header(head: str, lineno: int) -> None:
    found = sorted({character for character in head if character in _CONTROL})
    if not found:
        return
    shown = ", ".join(repr(character) for character in found[:5])
    raise SequenceError(
        f"line {lineno}: header holds control character(s) {shown}; a header is "
        "a line of printable text, and these would be interpreted by a terminal "
        "that printed it"
    )


def _split_header(head: str, lineno: int) -> tuple[str, str]:
    if not head:
        raise SequenceError(f"line {lineno}: header has no identifier")
    _check_header(head, lineno)
    fields = head.split(None, 1)
    return fields[0], fields[1] if len(fields) > 1 else ""


def parse_fasta(lines: Iterable[str]) -> Iterator[Record]:
    """Yield records from FASTA text."""
    identifier = ""
    description = ""
    chunks: list[str] = []
    started = False

    for lineno, raw in enumerate(lines, start=1):
        if lineno == 1:
            # A file exported on Windows may open with a byte-order mark, which
            # would otherwise make the first header unrecognisable.
            raw = raw.lstrip("\ufeff")
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if started:
                yield Record(identifier, description, "".join(chunks))
            identifier, description = _split_header(line[1:].strip(), lineno)
            chunks = []
            started = True
            continue
        if not started:
            raise SequenceError(f"line {lineno}: sequence data before any '>' header")
        residues = "".join(line.split())
        _check_residues(residues, lineno)
        chunks.append(residues.upper())

    if not started:
        raise SequenceError("input contains no FASTA records")
    yield Record(identifier, description, "".join(chunks))


def parse_fastq(lines: Iterable[str]) -> Iterator[Record]:
    """Yield records from four-line FASTQ input.

    Line-wrapped FASTQ is refused rather than guessed at: the format gives no
    reliable way to tell a wrapped quality line from the start of the next
    record, and a wrong guess would silently align the wrong residues.
    """
    block: list[str] = []
    first_lineno = 1
    records = 0

    for lineno, raw in enumerate(lines, start=1):
        if lineno == 1:
            raw = raw.lstrip("\ufeff")
        line = raw.rstrip("\r\n")
        if not block and not line.strip():
            continue
        if not block:
            first_lineno = lineno
        block.append(line)
        if len(block) < 4:
            continue
        yield _fastq_record(block, first_lineno)
        records += 1
        block = []

    if block:
        raise SequenceError(
            f"line {first_lineno}: truncated FASTQ record; expected 4 lines, "
            f"got {len(block)}"
        )
    if not records:
        raise SequenceError("input contains no FASTQ records")


def _fastq_record(block: list[str], lineno: int) -> Record:
    header, sequence, separator, quality = block
    if not header.startswith("@"):
        raise SequenceError(f"line {lineno}: FASTQ record does not begin with '@'")
    if not separator.startswith("+"):
        raise SequenceError(
            f"line {lineno + 2}: expected a '+' separator line; line-wrapped FASTQ "
            "is not supported"
        )
    identifier, description = _split_header(header[1:].strip(), lineno)
    residues = "".join(sequence.split())
    _check_residues(residues, lineno + 1)
    _check_quality(quality, lineno + 3)
    if len(residues) != len(quality):
        raise SequenceError(
            f"line {lineno + 3}: quality string is {len(quality)} characters for a "
            f"{len(residues)}-residue sequence"
        )
    return Record(identifier, description, residues.upper(), quality)


class _ReleasingTextWrapper(io.TextIOWrapper):
    """A text wrapper that lets go of its stream instead of closing it.

    Standard input belongs to the process, not to this reader. A plain
    ``TextIOWrapper`` around ``sys.stdin.buffer`` closes that buffer when it is
    closed or collected, so merely reading standard input once would break
    every later read of it in the same process - a second command in the same
    interpreter, or anything the caller does after an API call. Detaching
    flushes and severs the link, leaving the stream open.

    Every layer this reader stacked over the stream it does not own is released
    in turn, because each of those would close the one below it on its way out.
    """

    #: Layers built over the unowned stream, outermost first.
    release_also: tuple[IO[bytes], ...] = ()

    def close(self) -> None:
        for layer in (self, *self.release_also):
            try:
                layer.detach()  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                # Either nothing to detach from, or already detached, which is
                # the state this is trying to reach.
                pass


def _decompressed(binary: IO[bytes], release: bool = False) -> IO[str]:
    """Wrap a binary stream in text mode, transparently decompressing it.

    Args:
        binary: the stream to read.
        release: keep ``binary`` open when the wrapper is closed or collected,
            for a stream this package does not own.
    """
    created: list[IO[bytes]] = []
    if hasattr(binary, "peek"):
        buffered = binary
    else:
        buffered = io.BufferedReader(binary)  # type: ignore[arg-type]
        created.append(buffered)
    head = bytes(buffered.peek(6)[:6])  # type: ignore[attr-defined]
    stream: IO[bytes] = buffered
    for magic, opener in _MAGIC:
        if head.startswith(magic):
            # A decompressor built on a file object does not close that object
            # on its way out, so it needs no releasing of its own.
            stream = opener(buffered)
            break
    if not release:
        return io.TextIOWrapper(stream, encoding="utf-8-sig", newline=None)
    wrapper = _ReleasingTextWrapper(stream, encoding="utf-8-sig", newline=None)
    wrapper.release_also = tuple(created)
    return wrapper


def open_text(source: str | Path | IO[str]) -> IO[str]:
    """Open a path, '-' for standard input, or pass through a text handle."""
    if hasattr(source, "read"):
        return source  # type: ignore[return-value]
    if source == "-":
        return _decompressed(sys.stdin.buffer, release=True)
    path = Path(source)  # type: ignore[arg-type]
    if path.is_dir():
        raise IsADirectoryError(f"not a file: {source}")
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {source}")
    return _decompressed(path.open("rb"))


def decoded(lines: Iterable[str]) -> Iterator[str]:
    """Yield lines, reporting a corrupt or wrongly encoded stream as input error.

    Decompression and decoding both happen lazily, as lines are pulled, so a
    truncated gzip or a file saved as UTF-16 does not fail when it is opened:
    it fails in the middle of reading, as an EOFError or a UnicodeDecodeError
    that says nothing about sequence. A half-finished download is an ordinary
    event and deserves an ordinary error.

    The whole stream is wrapped in one ``try`` rather than guarding each line,
    because ``yield from`` hands the iteration straight to the underlying file
    object and so costs nothing per line.
    """
    try:
        yield from lines
    except EOFError as error:
        raise SequenceError(
            "compressed input is truncated: it ended before its end-of-stream marker"
        ) from error
    except (lzma.LZMAError, zlib.error, gzip.BadGzipFile) as error:
        raise SequenceError(f"compressed input is corrupt: {error}") from error
    except UnicodeDecodeError as error:
        byte = error.object[error.start]
        raise SequenceError(
            f"input is not UTF-8 text: byte {byte:#04x} at position {error.start} "
            "is not valid UTF-8; a file saved as UTF-16 or in a legacy code page "
            "has to be re-encoded"
        ) from error
    except OSError as error:
        # bzip2 reports a damaged stream as a plain OSError, and so does a disk
        # that has gone away mid-read. Neither is a Python traceback's
        # business: both are input the tool could not read.
        raise SequenceError(f"input cannot be read: {error}") from error


def detect_format(lines: Iterable[str]) -> tuple[str, Iterator[str]]:
    """Sniff FASTA or FASTQ from the first non-blank line.

    Returns the format and an iterator over the input with that line restored,
    so sniffing costs nothing and works on a pipe.
    """
    iterator = iter(lines)
    prefix: list[str] = []
    for raw in iterator:
        line = raw.lstrip("\ufeff").strip() if not prefix else raw.strip()
        prefix.append(raw)
        if not line:
            continue
        if line.startswith(">"):
            return "fasta", _restore(prefix, iterator)
        if line.startswith("@"):
            return "fastq", _restore(prefix, iterator)
        raise SequenceError(
            "input is neither FASTA nor FASTQ: the first line begins with "
            f"{line[:1]!r}, not '>' or '@'"
        )
    raise SequenceError("input is empty")


def _restore(prefix: list[str], rest: Iterator[str]) -> Iterator[str]:
    yield from prefix
    yield from rest


def read_records(
    source: str | Path | IO[str],
    input_format: str = "auto",
    limit: int = 0,
) -> Iterator[Record]:
    """Yield records from a path, handle or '-', sniffing the format by default.

    Args:
        source: path, open text handle, or '-' for standard input.
        input_format: 'auto', 'fasta' or 'fastq'.
        limit: stop after this many records; 0 reads everything.

    Raises:
        SequenceError: if the input cannot be parsed.
    """
    if input_format not in FORMATS:
        known = ", ".join(FORMATS)
        raise SequenceError(
            f"unknown input format: {input_format!r}; choose from {known}"
        )
    handle = open_text(source)
    # A handle this function opened is this function's to close, however the
    # iteration ends: exhausted, stopped by the limit, abandoned by the caller
    # or unwound by an exception. A handle passed in belongs to the caller, and
    # standard input is not ours to close at all, or a second command in the
    # same process would find it shut.
    ours = not hasattr(source, "read") and source != "-"
    try:
        lines: Iterable[str] = decoded(handle)
        if input_format == "auto":
            input_format, lines = detect_format(lines)
        parser = parse_fasta if input_format == "fasta" else parse_fastq
        for count, record in enumerate(parser(lines), start=1):
            yield record
            if limit and count >= limit:
                return
    finally:
        if ours:
            handle.close()


def read_all(
    source: str | Path | IO[str],
    input_format: str = "auto",
    limit: int = 0,
) -> list[Record]:
    """Read every record of an input, in file order.

    Raises:
        SequenceError: if the input cannot be parsed, or holds no records.
    """
    records = list(read_records(source, input_format, limit))
    if not records:
        raise SequenceError(f"{source_name(source)} holds no records")
    return records


def source_name(source: str | Path | IO[str]) -> str:
    """How an input is named in output."""
    if source == "-":
        return "standard input"
    if hasattr(source, "read"):
        return str(getattr(source, "name", "input"))
    return Path(source).name  # type: ignore[arg-type]
