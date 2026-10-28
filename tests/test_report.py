"""The HTML report: self-contained, deterministic, and true to its tables."""

from __future__ import annotations

import base64
import math
import re

import pytest

from align_pair import report
from align_pair.aligner import Gaps
from align_pair.analysis import Settings, analyse
from align_pair.reader import Record
from align_pair.report import cumulative, render_html
from align_pair.rows import COLUMN_COLUMNS, PAIR_COLUMNS, column_rows, pair_rows, render

HUMAN = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
MOUSE = "MVHLTDAEKAAVSGLWGKVNADEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNAK"


@pytest.fixture
def analysis():
    return analyse(
        [Record("human", "beta globin", HUMAN)],
        [Record("mouse", "beta globin", MOUSE)],
        Settings(),
        ["human.fa", "mouse.fa"],
    )


@pytest.fixture
def document(analysis):
    return render_html(analysis, "1.0.0test")


class TestSelfContained:
    def test_no_script_runs(self, document: str) -> None:
        assert "<script" not in document.lower()
        assert "javascript:" not in document.lower()
        assert "onclick" not in document.lower()

    def test_nothing_is_fetched_from_anywhere(self, document: str) -> None:
        """Every reference has to be a data URI or a fragment.

        A report that fetches a stylesheet is a report that looks wrong on a
        laptop with no network and broken once the host moves.
        """
        assert not re.search(r'(?:src|href)="(?!data:|#)', document)

    def test_the_style_travels_with_the_document(self, document: str) -> None:
        assert "<style>" in document
        assert "font-family" in document

    def test_it_is_one_html_document(self, document: str) -> None:
        assert document.startswith("<!DOCTYPE html>")
        assert document.rstrip().endswith("</html>")


class TestDeterminism:
    def test_two_renderings_agree_byte_for_byte(self, analysis) -> None:
        assert render_html(analysis, "1.0.0test") == render_html(analysis, "1.0.0test")

    def test_two_runs_of_the_same_input_agree(self) -> None:
        """Nothing in the document may depend on the clock or on a set's order."""
        first = analyse(
            [Record("human", "", HUMAN)], [Record("mouse", "", MOUSE)], Settings()
        )
        second = analyse(
            [Record("human", "", HUMAN)], [Record("mouse", "", MOUSE)], Settings()
        )
        assert render_html(first, "x") == render_html(second, "x")

    def test_no_timestamp_appears(self, document: str) -> None:
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b", document)


class TestContent:
    def test_the_title_names_its_subject(self, document: str) -> None:
        title = re.search(r"<title>(.*?)</title>", document).group(1)
        assert title == "human aligned with mouse"

    def test_several_pairs_are_titled_by_their_number(self) -> None:
        analysis = analyse(
            [Record("a", "", HUMAN), Record("b", "", MOUSE), Record("c", "", HUMAN)],
            None,
            Settings(),
        )
        document = render_html(analysis, "x")
        title = re.search(r"<title>(.*?)</title>", document).group(1)
        assert title == "3 alignments"

    def test_the_parameters_are_stated(self, document: str) -> None:
        assert "BLOSUM62" in document
        assert "10 to open, 0.5 to extend" in document
        assert "(L - 1)" in document

    def test_the_alignment_is_shown(self, document: str) -> None:
        assert 'pre class="view"' in document
        assert "MAAD" not in document
        assert "MVHLT" in document

    def test_every_kind_of_column_is_coloured(self, document: str) -> None:
        for kind in ("same", "like", "diff"):
            assert f'<span class="{kind}">' in document

    def test_the_marker_line_is_kept_beside_the_colours(self, document: str) -> None:
        """The document has to be readable in black and white.

        Colour alone would exclude a reader who cannot distinguish it and a
        printout.
        """
        assert '<span class="marks">' in document
        assert "identical residues" in document

    def test_a_qualification_is_shown(self) -> None:
        analysis = analyse(
            [Record("a", "", HUMAN)],
            [Record("b", "", MOUSE)],
            Settings(gaps=Gaps(0.0, 0.0)),
        )
        document = render_html(analysis, "x")
        assert 'class="caveat"' in document
        assert "cost nothing" in document

    def test_the_matrix_comments_are_carried(self, document: str) -> None:
        assert "1/2 Bit Units" in document

    def test_the_whole_matrix_is_available(self, document: str) -> None:
        assert "the whole table" in document

    def test_the_footer_names_the_version(self, document: str) -> None:
        assert "align-pair 1.0.0test" in document

    def test_the_gap_convention_is_stated(self, document: str) -> None:
        assert "EMBOSS needle and water" in document


class TestEmbeddedTables:
    def _embedded(self, document: str) -> list[str]:
        found = re.findall(
            r'href="data:text/tab-separated-values;base64,([^"]+)"', document
        )
        return [base64.b64decode(payload).decode("utf-8") for payload in found]

    def test_the_tables_are_embedded_for_download(self, document: str) -> None:
        assert len(self._embedded(document)) == 2

    def test_an_embedded_table_is_the_file_the_cli_would_write(
        self, analysis, document: str
    ) -> None:
        """A table downloaded from the report is the table --output writes.

        Both come from the same serialiser, and this asserts it byte for byte
        rather than trusting that they stayed in step.
        """
        embedded = self._embedded(document)
        assert render(PAIR_COLUMNS, pair_rows(analysis)) in embedded
        assert (
            render(COLUMN_COLUMNS, column_rows(analysis.pairs[0], analysis.matrix))
            in embedded
        )

    def test_a_table_too_large_is_named_rather_than_embedded(
        self, analysis, monkeypatch
    ) -> None:
        monkeypatch.setattr(report, "MAX_EMBEDDED", 10)
        document = render_html(analysis, "x")
        assert "too large to embed" in document
        assert "data:text/tab-separated-values" not in document


class TestChart:
    def test_the_running_score_ends_at_the_alignment_score(self, analysis) -> None:
        """The chart decomposes the score rather than illustrating it.

        If the last point were not the score, the picture would be of some
        other quantity with the score's name on it.
        """
        pair = analysis.pairs[0]
        track = cumulative(pair, analysis.matrix, analysis.settings.gaps)
        assert track[-1][1] == pytest.approx(pair.alignment.score)

    def test_the_track_has_a_point_for_every_column_and_a_start(
        self, analysis
    ) -> None:
        pair = analysis.pairs[0]
        track = cumulative(pair, analysis.matrix, analysis.settings.gaps)
        assert len(track) == pair.metrics.length + 1
        assert track[0] == (0, 0.0)

    def test_gaps_are_charged_where_they_fall(self) -> None:
        analysis = analyse(
            [Record("a", "", "ACGTACGTACGTACGTACGTACGT")],
            [Record("b", "", "ACGTACGTACGTACGTACGTACG")],
            Settings(matrix="simple", match=1.0, mismatch=-1.0, gaps=Gaps(2.0, 1.0)),
        )
        pair = analysis.pairs[0]
        track = cumulative(pair, analysis.matrix, analysis.settings.gaps)
        steps = [
            second - first
            for (_i, first), (_j, second) in zip(track, track[1:])
        ]
        assert math.isclose(min(steps), -2.0)

    @pytest.mark.parametrize(
        "first,second,mismatch",
        [
            ("A", "C", -100.0),
            ("ACGT", "AGT", -1.0),
            ("ACCCGT", "AGT", -1.0),
            ("AAAACCCC", "CCCCAAAA", -1.0),
            ("ACGTACGTAC", "ACGTTTTACGTAC", -1.0),
        ],
    )
    def test_the_track_ends_at_the_score_whatever_the_gaps(
        self, first: str, second: str, mismatch: float
    ) -> None:
        """Opposite-direction gaps each open, so each pays the opening penalty.

        A running score that charged only one of them would end somewhere other
        than the score, and the chart would be of a quantity with the score's
        name on it. 'A' against 'C' under a punishing mismatch is the case:
        the optimum is two single-residue gaps, not one of two.
        """
        analysis = analyse(
            [Record("a", "", first)],
            [Record("b", "", second)],
            Settings(matrix="simple", match=1.0, mismatch=mismatch, gaps=Gaps(2.0, 1.0)),
        )
        pair = analysis.pairs[0]
        track = cumulative(pair, analysis.matrix, analysis.settings.gaps)
        assert track[-1][1] == pytest.approx(pair.alignment.score), (
            pair.alignment.aligned_first,
            pair.alignment.aligned_second,
        )

    def test_the_chart_is_drawn_in_the_document(self, document: str) -> None:
        assert '<svg class="chart"' in document
        assert "running score along the alignment" in document

    def test_an_empty_alignment_draws_no_chart(self) -> None:
        analysis = analyse(
            [Record("a", "", "AAAAAAAAAAAAAAAAAAAAAAAA")],
            [Record("b", "", "CCCCCCCCCCCCCCCCCCCCCCCC")],
            Settings(mode="local", matrix="simple", mismatch=-5.0),
        )
        document = render_html(analysis, "x")
        assert "No residues are aligned." in document
        assert '<svg class="chart"' not in document
