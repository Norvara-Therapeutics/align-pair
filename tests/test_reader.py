"""Reading FASTA and FASTQ, and refusing what is not sequence."""

from __future__ import annotations

import bz2
import gc
import gzip
import io
import lzma
import sys

import pytest

from align_pair import reader
from align_pair.reader import Record, SequenceError


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestFasta:
    def test_one_record_is_read(self) -> None:
        records = list(reader.parse_fasta([">one a sequence", "ACGT"]))
        assert len(records) == 1
        assert records[0].identifier == "one"
        assert records[0].description == "a sequence"
        assert records[0].sequence == "ACGT"

    def test_several_records_are_read_in_order(self) -> None:
        records = list(reader.parse_fasta([">a", "AC", ">b", "GT"]))
        assert [record.identifier for record in records] == ["a", "b"]

    def test_wrapped_lines_are_joined(self) -> None:
        records = list(reader.parse_fasta([">a", "AC", "GT", "AA"]))
        assert records[0].sequence == "ACGTAA"

    def test_residues_are_uppercased(self) -> None:
        records = list(reader.parse_fasta([">a", "acgt"]))
        assert records[0].sequence == "ACGT"

    def test_whitespace_inside_a_line_is_removed(self) -> None:
        records = list(reader.parse_fasta([">a", "AC GT\tAA"]))
        assert records[0].sequence == "ACGTAA"

    def test_blank_lines_are_skipped(self) -> None:
        records = list(reader.parse_fasta([">a", "", "ACGT", ""]))
        assert records[0].sequence == "ACGT"

    def test_a_byte_order_mark_does_not_hide_the_header(self) -> None:
        records = list(reader.parse_fasta(["\ufeff>a", "ACGT"]))
        assert records[0].identifier == "a"

    def test_a_record_with_no_residues_is_still_a_record(self) -> None:
        records = list(reader.parse_fasta([">a", ">b", "AC"]))
        assert records[0].sequence == ""
        assert len(records) == 2

    def test_sequence_before_a_header_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="before any '>' header"):
            list(reader.parse_fasta(["ACGT"]))

    def test_an_empty_input_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="no FASTA records"):
            list(reader.parse_fasta([]))

    def test_a_header_with_no_identifier_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="no identifier"):
            list(reader.parse_fasta([">", "ACGT"]))

    def test_data_that_is_not_sequence_is_refused(self) -> None:
        """A saved error page or a column of numbers is not a sequence.

        Absorbing it as residues and aligning it is worse than an error.
        """
        with pytest.raises(SequenceError, match="not sequence data"):
            list(reader.parse_fasta([">a", "12345"]))

    def test_the_offending_characters_are_named(self) -> None:
        with pytest.raises(SequenceError, match="'1'"):
            list(reader.parse_fasta([">a", "AC1GT"]))

    def test_the_line_number_is_named(self) -> None:
        with pytest.raises(SequenceError, match="line 4"):
            list(reader.parse_fasta([">a", "ACGT", "ACGT", "AC!GT"]))

    @pytest.mark.parametrize(
        "header",
        [
            ">ev\x1b[2Jil",
            ">name \x1b]0;title\x07",
            ">name\x07",
            ">na\x00me",
            ">name \x7f",
        ],
    )
    def test_a_control_character_in_a_header_is_refused(self, header: str) -> None:
        """A header is a line of printable text.

        An alignment summary names its records, so an identifier carrying an
        ANSI erase or an OSC title-set would be interpreted by the terminal of
        whoever printed it.
        """
        with pytest.raises(SequenceError, match="control character"):
            list(reader.parse_fasta([header, "ACGT"]))

    def test_the_offending_character_is_named(self) -> None:
        with pytest.raises(SequenceError, match=r"\\x1b"):
            list(reader.parse_fasta([">a\x1b[2J", "ACGT"]))

    def test_an_ordinary_header_is_untouched(self) -> None:
        records = list(reader.parse_fasta([">NP_000509.1 hemoglobin [Homo sapiens]", "AC"]))
        assert records[0].identifier == "NP_000509.1"
        assert records[0].description == "hemoglobin [Homo sapiens]"

    def test_a_header_is_the_identifier_and_description(self) -> None:
        records = list(reader.parse_fasta([">one some words here", "AC"]))
        assert records[0].header == "one some words here"

    def test_a_record_knows_its_length(self) -> None:
        records = list(reader.parse_fasta([">a", "ACGT"]))
        assert len(records[0]) == 4


class TestFastq:
    BLOCK = ["@read one", "ACGT", "+", "IIII"]

    def test_a_record_is_read(self) -> None:
        records = list(reader.parse_fastq(self.BLOCK))
        assert records[0].identifier == "read"
        assert records[0].sequence == "ACGT"
        assert records[0].quality == "IIII"

    def test_a_truncated_record_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="truncated FASTQ"):
            list(reader.parse_fastq(self.BLOCK[:3]))

    def test_a_missing_at_sign_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="does not begin with '@'"):
            list(reader.parse_fastq(["read", "ACGT", "+", "IIII"]))

    def test_a_missing_separator_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="line-wrapped FASTQ"):
            list(reader.parse_fastq(["@read", "ACGT", "ACGT", "IIII"]))

    def test_a_quality_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="4-residue sequence"):
            list(reader.parse_fastq(["@read", "ACGT", "+", "III"]))

    def test_a_quality_outside_printable_ascii_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="not a quality string"):
            list(reader.parse_fastq(["@read", "ACGT", "+", "II\u00e9I"]))

    def test_an_empty_input_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="no FASTQ records"):
            list(reader.parse_fastq([]))


class TestDetection:
    def test_fasta_is_recognised(self) -> None:
        kind, lines = reader.detect_format([">a", "ACGT"])
        assert kind == "fasta"
        assert list(lines) == [">a", "ACGT"]

    def test_fastq_is_recognised(self) -> None:
        kind, lines = reader.detect_format(["@a", "ACGT", "+", "IIII"])
        assert kind == "fastq"
        assert len(list(lines)) == 4

    def test_the_sniffed_line_is_put_back(self) -> None:
        """Sniffing must cost nothing, because the input may be a pipe."""
        _kind, lines = reader.detect_format(["", ">a", "ACGT"])
        assert list(lines) == ["", ">a", "ACGT"]

    def test_anything_else_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="neither FASTA nor FASTQ"):
            reader.detect_format(["LOCUS  something"])

    def test_an_empty_stream_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="input is empty"):
            reader.detect_format([])


class TestFiles:
    def test_a_path_is_read(self, tmp_path) -> None:
        path = write(tmp_path, "a.fa", ">a\nACGT\n")
        assert reader.read_all(path)[0].sequence == "ACGT"

    def test_a_handle_is_read(self) -> None:
        records = reader.read_all(io.StringIO(">a\nACGT\n"))
        assert records[0].sequence == "ACGT"

    def test_a_missing_file_is_refused(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="no such file"):
            reader.read_all(tmp_path / "absent.fa")

    def test_a_directory_is_refused(self, tmp_path) -> None:
        with pytest.raises(IsADirectoryError, match="not a file"):
            reader.read_all(tmp_path)

    def test_the_limit_stops_reading(self, tmp_path) -> None:
        path = write(tmp_path, "a.fa", ">a\nAC\n>b\nGT\n>c\nAA\n")
        assert len(reader.read_all(path, limit=2)) == 2

    def test_an_unknown_format_is_refused(self, tmp_path) -> None:
        path = write(tmp_path, "a.fa", ">a\nACGT\n")
        with pytest.raises(SequenceError, match="unknown input format"):
            reader.read_all(path, "genbank")

    @pytest.mark.parametrize(
        "compress,suffix",
        [(gzip.compress, ".gz"), (bz2.compress, ".bz2"), (lzma.compress, ".xz")],
    )
    def test_compression_is_recognised_from_the_bytes(
        self, tmp_path, compress, suffix
    ) -> None:
        """The file name is not consulted, so a stripped suffix still works."""
        path = tmp_path / f"a.fa{suffix}"
        path.write_bytes(compress(b">a\nACGT\n"))
        assert reader.read_all(path)[0].sequence == "ACGT"
        plain = tmp_path / "misnamed.fa"
        plain.write_bytes(compress(b">a\nACGT\n"))
        assert reader.read_all(plain)[0].sequence == "ACGT"

    def test_a_truncated_archive_is_reported_as_input_error(self, tmp_path) -> None:
        path = tmp_path / "a.fa.gz"
        path.write_bytes(gzip.compress(b">a\n" + b"ACGT" * 2000 + b"\n")[:-40])
        with pytest.raises(SequenceError, match="truncated|corrupt"):
            reader.read_all(path)

    def test_a_file_in_the_wrong_encoding_is_reported(self, tmp_path) -> None:
        path = tmp_path / "a.fa"
        path.write_bytes(">a\nACGT\n".encode("utf-16"))
        with pytest.raises(SequenceError, match="not UTF-8 text"):
            reader.read_all(path)

    def test_an_input_with_no_records_is_refused(self, tmp_path) -> None:
        path = write(tmp_path, "a.fa", "\n\n")
        with pytest.raises(SequenceError):
            reader.read_all(path)

    def test_a_handle_the_caller_opened_is_not_closed(self, tmp_path) -> None:
        """A handle passed in belongs to the caller."""
        path = write(tmp_path, "a.fa", ">a\nACGT\n")
        handle = path.open("r", encoding="utf-8")
        reader.read_all(handle)
        assert not handle.closed
        handle.close()

    def test_a_handle_this_package_opened_is_closed(self, tmp_path, monkeypatch) -> None:
        path = write(tmp_path, "a.fa", ">a\nAC\n>b\nGT\n")
        opened = []
        original = reader.open_text

        def spy(source):
            handle = original(source)
            opened.append(handle)
            return handle

        monkeypatch.setattr(reader, "open_text", spy)
        # Abandoning the iterator part-way must still close the file.
        for _record in reader.read_records(path):
            break
        assert opened and all(handle.closed for handle in opened)


class TestStandardInput:
    """Standard input belongs to the process, not to this reader.

    A plain text wrapper around ``sys.stdin.buffer`` closes that buffer when it
    is collected, so reading standard input once would break every later read
    of it in the same process.
    """

    class Fake:
        def __init__(self, data: bytes, buffered: bool) -> None:
            self.base = io.BytesIO(data)
            self.buffer = io.BufferedReader(self.base) if buffered else self.base

    @pytest.mark.parametrize("buffered", [True, False])
    def test_reading_it_leaves_it_open(self, buffered: bool, monkeypatch) -> None:
        fake = self.Fake(b">a\nACGT\n", buffered)
        monkeypatch.setattr(sys, "stdin", fake)
        records = reader.read_all("-")
        assert records[0].sequence == "ACGT"
        del records
        gc.collect()
        assert not fake.buffer.closed
        assert not fake.base.closed

    @pytest.mark.parametrize("buffered", [True, False])
    def test_a_compressed_stream_leaves_it_open(
        self, buffered: bool, monkeypatch
    ) -> None:
        """The decompressor is another layer that would close what is below it."""
        fake = self.Fake(gzip.compress(b">a\nACGT\n"), buffered)
        monkeypatch.setattr(sys, "stdin", fake)
        assert reader.read_all("-")[0].sequence == "ACGT"
        gc.collect()
        assert not fake.base.closed

    def test_it_can_be_read_twice_in_one_process(self, monkeypatch) -> None:
        """Two commands in one interpreter, which is what a test run is."""
        fake = self.Fake(b">a\nAC\n>b\nGT\n", True)
        monkeypatch.setattr(sys, "stdin", fake)
        assert len(reader.read_all("-")) == 2
        fake.buffer.seek(0)
        assert len(reader.read_all("-")) == 2

    def test_a_file_this_package_opened_is_still_closed(self, tmp_path) -> None:
        """Releasing is for streams the package does not own, not for files."""
        path = write(tmp_path, "a.fa", ">a\nACGT\n")
        handle = reader.open_text(path)
        handle.close()
        assert handle.closed


class TestNames:
    def test_a_path_is_named_by_its_file_name(self) -> None:
        assert reader.source_name("/tmp/some/where.fa") == "where.fa"

    def test_standard_input_is_named(self) -> None:
        assert reader.source_name("-") == "standard input"

    def test_a_handle_is_named_by_its_name(self, tmp_path) -> None:
        path = write(tmp_path, "a.fa", ">a\nAC\n")
        with path.open() as handle:
            assert reader.source_name(handle).endswith("a.fa")


class TestRecord:
    def test_a_header_without_a_description_is_just_the_identifier(self) -> None:
        assert Record("a", "", "AC").header == "a"

    def test_the_residue_set_covers_every_letter(self) -> None:
        assert set("ACGTURYSWKMBDHVNX*-") <= reader.VALID_RESIDUES
