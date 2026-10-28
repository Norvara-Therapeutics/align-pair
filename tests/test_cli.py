"""The command line: what it writes, what it refuses, and what it never eats."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from align_pair.cli import build_parser, main

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

HUMAN = ">human beta globin\nMVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK\n"
MOUSE = ">mouse beta globin\nMVHLTDAEKAAVSGLWGKVNADEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNAK\n"
DNA_ONE = ">one\nACGTACGTACGTACGTACGTACGT\n"
DNA_TWO = ">two\nACGTACGTACGTACGTACGTACGA\n"


@pytest.fixture
def pair(tmp_path):
    first = tmp_path / "human.fa"
    second = tmp_path / "mouse.fa"
    first.write_text(HUMAN, encoding="utf-8")
    second.write_text(MOUSE, encoding="utf-8")
    return first, second


def invoke(arguments, capsys):
    code = main(arguments)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestSummary:
    def test_a_pair_is_aligned_and_summarised(self, pair, capsys) -> None:
        code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1])], capsys
        )
        assert code == 0
        assert "> human vs mouse" in out
        assert "BLOSUM62" in out
        assert "identity" in out

    def test_the_alignment_is_shown_by_default(self, pair, capsys) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1])], capsys
        )
        assert "MVHLT" in out
        assert "|||||" in out

    def test_no_view_keeps_the_metrics_and_drops_the_alignment(
        self, pair, capsys
    ) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--no-view"], capsys
        )
        assert "identity" in out
        assert "|||||" not in out

    def test_every_mode_runs(self, pair, capsys) -> None:
        for mode in ("global", "glocal", "overlap", "local"):
            code, out, _err = invoke(
                ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-m", mode], capsys
            )
            assert code == 0
            assert mode in out

    def test_one_input_aligns_its_records_against_each_other(
        self, tmp_path, capsys
    ) -> None:
        path = tmp_path / "both.fa"
        path.write_text(HUMAN + MOUSE, encoding="utf-8")
        code, out, _err = invoke(["align", "-a", str(path), "--no-view"], capsys)
        assert code == 0
        assert out.count("> ") == 1

    def test_a_single_record_with_no_partner_is_refused(
        self, pair, capsys
    ) -> None:
        code, _out, err = invoke(["align", "-a", str(pair[0])], capsys)
        assert code == 2
        assert "nothing to align it to" in err


class TestFormats:
    def test_tsv_writes_one_row_per_pair(self, pair, capsys) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-f", "tsv"], capsys
        )
        lines = out.strip().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("first\tsecond\t")

    def test_json_carries_the_alignment_and_the_qualifications(
        self, pair, capsys
    ) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-f", "json"], capsys
        )
        document = json.loads(out)
        assert document["matrix"]["name"] == "BLOSUM62"
        assert document["gap_model"] == "open + (length - 1) * extend"
        pair_document = document["pairs"][0]
        assert pair_document["alignment"]["first"].startswith("MVHLT")
        assert "markers" in pair_document["alignment"]
        assert "qualifications" in pair_document

    def test_the_json_is_strict(self, pair, capsys) -> None:
        """No NaN or Infinity, which are not JSON and break strict parsers."""
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-f", "json"], capsys
        )
        json.loads(out, parse_constant=_refuse)

    def test_none_writes_nothing(self, pair, capsys) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-f", "none"], capsys
        )
        assert out == ""


class TestFileOutputs:
    def test_the_summary_table_is_written(self, pair, tmp_path, capsys) -> None:
        out_path = tmp_path / "pairs.tsv"
        invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(out_path),
                "-f",
                "none",
            ],
            capsys,
        )
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_the_column_table_is_written(self, pair, tmp_path, capsys) -> None:
        out_path = tmp_path / "columns.tsv"
        invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "--columns",
                str(out_path),
                "-f",
                "none",
            ],
            capsys,
        )
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 61

    def test_the_aligned_pair_is_written_as_fasta(
        self, pair, tmp_path, capsys
    ) -> None:
        out_path = tmp_path / "aligned.fa"
        invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "--alignment",
                str(out_path),
                "-f",
                "none",
            ],
            capsys,
        )
        text = out_path.read_text()
        assert text.startswith(">human beta globin\n")
        assert ">mouse" in text

    def test_the_report_is_written(self, pair, tmp_path, capsys) -> None:
        out_path = tmp_path / "report.html"
        invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "--report",
                str(out_path),
                "-f",
                "none",
            ],
            capsys,
        )
        assert out_path.read_text().startswith("<!DOCTYPE html>")

    def test_an_output_may_go_to_standard_output(self, pair, capsys) -> None:
        code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-o", "-", "-f", "none"],
            capsys,
        )
        assert code == 0
        assert out.startswith("first\tsecond\t")

    def test_several_outputs_are_written_in_one_pass(
        self, pair, tmp_path, capsys
    ) -> None:
        invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(tmp_path / "a.tsv"),
                "--columns",
                str(tmp_path / "b.tsv"),
                "--report",
                str(tmp_path / "c.html"),
                "-f",
                "none",
            ],
            capsys,
        )
        for name in ("a.tsv", "b.tsv", "c.html"):
            assert (tmp_path / name).exists()


class TestPathGuards:
    def test_an_output_naming_an_input_is_refused(self, pair, capsys) -> None:
        """Writing it would truncate the file being read."""
        before = pair[0].read_bytes()
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-o", str(pair[0])],
            capsys,
        )
        assert code == 2
        assert "is the input" in err
        assert pair[0].read_bytes() == before

    def test_a_hard_link_to_an_input_is_the_input(
        self, pair, tmp_path, capsys
    ) -> None:
        """A hard link has a different name and the same file behind it.

        Comparing names alone would let the run destroy what it is reading.
        """
        link = tmp_path / "linked.fa"
        os.link(pair[0], link)
        before = pair[0].read_bytes()
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-o", str(link)], capsys
        )
        assert code == 2
        assert "is the input" in err
        assert pair[0].read_bytes() == before

    def test_a_symlink_to_an_input_is_the_input(
        self, pair, tmp_path, capsys
    ) -> None:
        link = tmp_path / "symlinked.fa"
        link.symlink_to(pair[1])
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--columns", str(link)],
            capsys,
        )
        assert code == 2
        assert "is the input" in err

    def test_a_matrix_file_counts_as_an_input(self, pair, tmp_path, capsys) -> None:
        """The matrix is read too, so an output aimed at it would destroy it."""
        matrix = tmp_path / "custom.mat"
        matrix.write_text("   A  C\nA  2 -1\nC -1  2\n", encoding="utf-8")
        before = matrix.read_bytes()
        code, _out, err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "--matrix",
                str(matrix),
                "-o",
                str(matrix),
            ],
            capsys,
        )
        assert code == 2
        assert "is the input" in err
        assert matrix.read_bytes() == before

    def test_two_outputs_naming_one_file_are_refused(
        self, pair, tmp_path, capsys
    ) -> None:
        """The second would silently replace the first."""
        target = tmp_path / "both.tsv"
        code, _out, err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(target),
                "--columns",
                str(target),
            ],
            capsys,
        )
        assert code == 2
        assert "two outputs name the same file" in err
        assert not target.exists()

    def test_two_spellings_of_one_file_are_one_file(
        self, pair, tmp_path, capsys
    ) -> None:
        target = tmp_path / "both.tsv"
        code, _out, err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(target),
                "--columns",
                str(tmp_path / "." / "both.tsv"),
            ],
            capsys,
        )
        assert code == 2
        assert "two outputs name the same file" in err

    def test_two_outputs_cannot_share_standard_output(
        self, pair, capsys
    ) -> None:
        code, _out, err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                "-",
                "--columns",
                "-",
            ],
            capsys,
        )
        assert code == 2
        assert "cannot share standard output" in err

    def test_a_summary_and_an_output_cannot_share_standard_output(
        self, pair, capsys
    ) -> None:
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-o", "-"], capsys
        )
        assert code == 2
        assert "--format none" in err

    def test_a_missing_input_is_refused(self, pair, tmp_path, capsys) -> None:
        code, _out, err = invoke(
            ["align", "-a", str(tmp_path / "absent.fa"), "-b", str(pair[1])], capsys
        )
        assert code == 2
        assert "no such file" in err

    def test_an_output_in_a_missing_directory_is_refused(
        self, pair, tmp_path, capsys
    ) -> None:
        code, _out, err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(tmp_path / "nowhere" / "a.tsv"),
            ],
            capsys,
        )
        assert code == 2
        assert "no such directory" in err

    def test_nothing_is_written_when_the_guard_refuses(
        self, pair, tmp_path, capsys
    ) -> None:
        """A run that cannot finish does not start."""
        good = tmp_path / "good.tsv"
        code, _out, _err = invoke(
            [
                "align",
                "-a",
                str(pair[0]),
                "-b",
                str(pair[1]),
                "-o",
                str(good),
                "--report",
                str(pair[0]),
            ],
            capsys,
        )
        assert code == 2
        assert not good.exists()


class TestPartialOutput:
    """A run that cannot be written does not write anything.

    Every destination is opened before any is written, and if one cannot be,
    the files this run created are removed again. Checking that a path looks
    writable is not the same as finding out: a name too long, a directory with
    no write permission or a full disk is only discovered on open.
    """

    def test_an_unwritable_later_destination_leaves_nothing_behind(
        self, pair, tmp_path, capsys
    ) -> None:
        locked = tmp_path / "locked"
        locked.mkdir()
        os.chmod(locked, stat.S_IREAD | stat.S_IEXEC)
        try:
            first = tmp_path / "first.tsv"
            code, _out, err = invoke(
                [
                    "align", "-a", str(pair[0]), "-b", str(pair[1]),
                    "-o", str(first), "--columns", str(locked / "second.tsv"),
                    "-f", "none",
                ],
                capsys,
            )
            assert code == 2
            assert "cannot write" in err
            assert not first.exists()
        finally:
            os.chmod(locked, stat.S_IRWXU)

    def test_an_overlong_name_exits_rather_than_raising(
        self, pair, tmp_path, capsys
    ) -> None:
        code, _out, err = invoke(
            [
                "align", "-a", str(pair[0]), "-b", str(pair[1]),
                "-o", str(tmp_path / ("x" * 300 + ".tsv")), "-f", "none",
            ],
            capsys,
        )
        assert code == 2
        assert "cannot write" in err

    def test_a_file_that_already_existed_is_not_deleted(
        self, pair, tmp_path, capsys
    ) -> None:
        """Removing something the run did not create would be worse."""
        locked = tmp_path / "locked"
        locked.mkdir()
        os.chmod(locked, stat.S_IREAD | stat.S_IEXEC)
        try:
            existing = tmp_path / "already.tsv"
            existing.write_text("mine\n", encoding="utf-8")
            code, _out, _err = invoke(
                [
                    "align", "-a", str(pair[0]), "-b", str(pair[1]),
                    "-o", str(existing), "--columns", str(locked / "second.tsv"),
                    "-f", "none",
                ],
                capsys,
            )
            assert code == 2
            assert existing.exists()
        finally:
            os.chmod(locked, stat.S_IRWXU)

    def test_a_writable_run_still_writes_everything(
        self, pair, tmp_path, capsys
    ) -> None:
        invoke(
            [
                "align", "-a", str(pair[0]), "-b", str(pair[1]),
                "-o", str(tmp_path / "a.tsv"),
                "--columns", str(tmp_path / "b.tsv"),
                "--alignment", str(tmp_path / "c.fa"),
                "--report", str(tmp_path / "d.html"),
                "-f", "none",
            ],
            capsys,
        )
        for name in ("a.tsv", "b.tsv", "c.fa", "d.html"):
            assert (tmp_path / name).read_text()


class TestMinusStrandOutputs:
    """A data file has no qualifications beside it, so it has to carry its own.

    A reverse-complemented row written under the record's own header is the
    record's identifier against residues the record does not contain.
    """

    @pytest.fixture
    def flipped(self, tmp_path):
        from align_pair.alphabet import DNA, reverse_complement

        query = tmp_path / "q.fa"
        query.write_text(">query\n" + DNA_ONE.split("\n")[1] + "\n", encoding="utf-8")
        target = tmp_path / "t.fa"
        target.write_text(
            ">target some description\n"
            + reverse_complement(DNA_ONE.split("\n")[1], DNA)
            + "\n",
            encoding="utf-8",
        )
        return query, target

    def test_the_aligned_fasta_marks_the_reverse_complement(
        self, flipped, tmp_path, capsys
    ) -> None:
        out = tmp_path / "aln.fa"
        invoke(
            [
                "align", "-a", str(flipped[0]), "-b", str(flipped[1]),
                "--strand", "minus", "--alignment", str(out), "-f", "none",
            ],
            capsys,
        )
        lines = out.read_text().splitlines()
        assert lines[0] == ">query"
        assert lines[2].startswith(">target reverse complement")
        assert "some description" in lines[2]

    def test_the_plus_strand_header_is_left_alone(
        self, flipped, tmp_path, capsys
    ) -> None:
        out = tmp_path / "aln.fa"
        invoke(
            [
                "align", "-a", str(flipped[0]), "-b", str(flipped[1]),
                "--alignment", str(out), "-f", "none",
            ],
            capsys,
        )
        assert ">target some description" in out.read_text()

    def test_the_column_table_names_the_strand(
        self, flipped, tmp_path, capsys
    ) -> None:
        out = tmp_path / "columns.tsv"
        invoke(
            [
                "align", "-a", str(flipped[0]), "-b", str(flipped[1]),
                "--strand", "minus", "--columns", str(out), "-f", "none",
            ],
            capsys,
        )
        lines = out.read_text().splitlines()
        assert "strand" in lines[0].split("\t")
        column = lines[0].split("\t").index("strand")
        assert {row.split("\t")[column] for row in lines[1:]} == {"minus"}

    def test_the_column_positions_count_along_the_record(
        self, flipped, tmp_path, capsys
    ) -> None:
        out = tmp_path / "columns.tsv"
        invoke(
            [
                "align", "-a", str(flipped[0]), "-b", str(flipped[1]),
                "--strand", "minus", "--columns", str(out), "-f", "none",
            ],
            capsys,
        )
        lines = out.read_text().splitlines()
        header = lines[0].split("\t")
        index = header.index("position_second")
        positions = [int(row.split("\t")[index]) for row in lines[1:]]
        assert positions == sorted(positions, reverse=True)


class TestOptions:
    def test_a_named_matrix_is_used(self, pair, capsys) -> None:
        _code, out, _err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--matrix", "PAM250",
             "--no-view"],
            capsys,
        )
        assert "PAM250" in out

    def test_a_matrix_file_is_read(self, tmp_path, capsys) -> None:
        first = tmp_path / "a.fa"
        second = tmp_path / "b.fa"
        first.write_text(DNA_ONE, encoding="utf-8")
        second.write_text(DNA_TWO, encoding="utf-8")
        matrix = tmp_path / "custom.mat"
        matrix.write_text(
            "   A  C  G  T\nA  2 -1 -1 -1\nC -1  2 -1 -1\nG -1 -1  2 -1\nT -1 -1 -1  2\n",
            encoding="utf-8",
        )
        code, out, _err = invoke(
            ["align", "-a", str(first), "-b", str(second), "--matrix", str(matrix),
             "--no-view"],
            capsys,
        )
        assert code == 0
        assert "custom.mat" in out

    def test_simple_takes_the_match_and_mismatch(self, tmp_path, capsys) -> None:
        first = tmp_path / "a.fa"
        second = tmp_path / "b.fa"
        first.write_text(DNA_ONE, encoding="utf-8")
        second.write_text(DNA_TWO, encoding="utf-8")
        _code, out, _err = invoke(
            [
                "align", "-a", str(first), "-b", str(second),
                "--matrix", "simple", "--match", "1", "--mismatch", "-1",
                "-f", "json",
            ],
            capsys,
        )
        document = json.loads(out)
        assert document["pairs"][0]["score"] == 22.0

    def test_match_without_simple_is_reported_as_ineffective(
        self, pair, capsys
    ) -> None:
        """No option is silently ignored."""
        _code, _out, err = invoke(
            [
                "align", "-a", str(pair[0]), "-b", str(pair[1]),
                "--matrix", "BLOSUM62", "--match", "9", "--no-view",
            ],
            capsys,
        )
        assert "--match has no effect" in err

    def test_no_view_without_the_table_is_reported_as_ineffective(
        self, pair, capsys
    ) -> None:
        _code, _out, err = invoke(
            [
                "align", "-a", str(pair[0]), "-b", str(pair[1]),
                "--no-view", "-f", "tsv",
            ],
            capsys,
        )
        assert "--no-view has no effect" in err

    def test_a_band_is_refused_outside_global_mode(self, pair, capsys) -> None:
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "-m", "local",
             "--band", "5"],
            capsys,
        )
        assert code == 2
        assert "only defined for global" in err

    def test_a_negative_gap_penalty_is_refused(self, pair, capsys) -> None:
        with pytest.raises(SystemExit):
            main(
                ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--gap-open", "-1"]
            )

    def test_a_penalty_that_is_not_a_number_is_refused(self, pair) -> None:
        with pytest.raises(SystemExit):
            main(
                ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--gap-open", "nan"]
            )

    def test_a_score_that_is_not_a_number_is_refused(self, pair) -> None:
        with pytest.raises(SystemExit):
            main(["align", "-a", str(pair[0]), "-b", str(pair[1]), "--match", "inf"])

    def test_the_memory_limit_refuses_a_large_pair(self, tmp_path, capsys) -> None:
        path = tmp_path / "big.fa"
        path.write_text(">big\n" + "ACGT" * 2000 + "\n", encoding="utf-8")
        other = tmp_path / "big2.fa"
        other.write_text(">big2\n" + "ACGT" * 2000 + "\n", encoding="utf-8")
        code, _out, err = invoke(
            ["align", "-a", str(path), "-b", str(other), "--max-memory", "1M"],
            capsys,
        )
        assert code == 2
        assert "--max-memory" in err

    @pytest.mark.parametrize(
        "written,expected",
        [("500", 500), ("2K", 2_000), ("10M", 10_000_000), ("1G", 1_000_000_000)],
    )
    def test_a_size_takes_a_suffix(self, written: str, expected: int) -> None:
        from align_pair.cli import _bytes_option

        assert _bytes_option(written) == expected

    def test_a_size_that_is_not_a_number_is_refused(self, pair) -> None:
        with pytest.raises(SystemExit):
            main(
                ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--max-memory", "big"]
            )

    def test_the_cell_limit_refuses_a_large_pair(self, pair, capsys) -> None:
        code, _out, err = invoke(
            ["align", "-a", str(pair[0]), "-b", str(pair[1]), "--max-cells", "10"],
            capsys,
        )
        assert code == 2
        assert "--max-cells" in err

    def test_degap_realigns_an_aligned_record(self, tmp_path, capsys) -> None:
        first = tmp_path / "a.fa"
        second = tmp_path / "b.fa"
        first.write_text(">a\nMVHLT--PEEKSAVTALWGKVNVDEV\n", encoding="utf-8")
        second.write_text(">b\nMVHLTDAEKAAVSGLWGKVNADEV\n", encoding="utf-8")
        code, _out, err = invoke(
            ["align", "-a", str(first), "-b", str(second), "--no-view"], capsys
        )
        assert code == 2
        assert "already aligned" in err
        code, out, _err = invoke(
            ["align", "-a", str(first), "-b", str(second), "--degap", "--no-view"],
            capsys,
        )
        assert code == 0
        assert "gap character" in out

    def test_the_limit_reads_only_the_first_records(self, tmp_path, capsys) -> None:
        path = tmp_path / "many.fa"
        path.write_text(HUMAN + MOUSE + HUMAN, encoding="utf-8")
        _code, out, _err = invoke(
            ["align", "-a", str(path), "--limit", "2", "--no-view"], capsys
        )
        assert out.count("> ") == 1


class TestHeaderSafety:
    def test_a_header_with_escapes_is_refused_before_it_is_printed(
        self, tmp_path, capsys
    ) -> None:
        """An alignment summary names its records.

        A header carrying an ANSI erase would be interpreted by the terminal
        that printed it, so the reader refuses it rather than the printer
        stripping it.
        """
        evil = tmp_path / "evil.fa"
        evil.write_bytes(b">ev\x1b[2Jil\nACGTACGTACGTACGTACGTACGT\n")
        plain = tmp_path / "plain.fa"
        plain.write_text(DNA_TWO, encoding="utf-8")
        code, out, err = invoke(
            ["align", "-a", str(evil), "-b", str(plain)], capsys
        )
        assert code == 2
        assert "\x1b" not in out
        assert "\x1b" not in err
        assert "control character" in err


class TestMatricesCommand:
    def test_the_built_ins_are_listed(self, capsys) -> None:
        code, out, _err = invoke(["matrices"], capsys)
        assert code == 0
        assert "BLOSUM62" in out
        assert "NUC.4.4" in out

    def test_one_matrix_is_printed_with_its_table(self, capsys) -> None:
        code, out, _err = invoke(["matrices", "BLOSUM62"], capsys)
        assert code == 0
        assert "1/2 Bit Units" in out
        assert "  W " in out

    def test_a_matrix_is_available_as_json(self, capsys) -> None:
        _code, out, _err = invoke(["matrices", "NUC.4.4", "-f", "json"], capsys)
        document = json.loads(out)
        assert document["scores"]["A"]["A"] == 5
        assert document["symmetric"] is True

    def test_a_matrix_is_available_as_tsv(self, capsys) -> None:
        _code, out, _err = invoke(["matrices", "BLOSUM62", "-f", "tsv"], capsys)
        assert out.splitlines()[0].startswith("\tA\tR\tN\t")

    def test_the_listing_is_available_as_tsv_and_json(self, capsys) -> None:
        _code, out, _err = invoke(["matrices", "-f", "tsv"], capsys)
        assert out.splitlines()[0] == "name\tscores\tresidues\trange\tsummary"
        _code, out, _err = invoke(["matrices", "-f", "json"], capsys)
        assert len(json.loads(out)) == 8

    def test_an_unknown_matrix_is_refused(self, capsys) -> None:
        code, _out, err = invoke(["matrices", "NOTAMATRIX"], capsys)
        assert code == 2
        assert "unknown matrix" in err


class TestParser:
    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_the_first_input_is_required(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["align"])

    def test_the_version_is_available(self, capsys) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["--version"])
        assert caught.value.code == 0
        assert "align-pair" in capsys.readouterr().out


class TestInstalledEntryPoints:
    def test_the_module_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "align_pair", "--version"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert result.returncode == 0
        assert "align-pair" in result.stdout

    def test_standard_input_is_read(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "align_pair",
                "align",
                "-a",
                "-",
                "-b",
                str(EXAMPLES / "hbb_mouse.fasta"),
                "--no-view",
            ],
            input=(EXAMPLES / "hbb_human.fasta").read_text(),
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        assert result.returncode == 0, result.stderr
        assert "NP_000509.1 vs NP_032246.2" in result.stdout


def _refuse(value):
    raise AssertionError(f"the JSON holds {value}, which is not JSON")
