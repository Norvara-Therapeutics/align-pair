"""The substitution matrices: the built-in tables, user files, and lookup."""

from __future__ import annotations

import math

import pytest

from align_pair import matrices, matrix_data
from align_pair.alphabet import DNA, PROTEIN, RNA
from align_pair.matrices import Matrix, MatrixError


class TestBuiltIn:
    def test_every_named_matrix_loads(self) -> None:
        for name in matrices.BUILT_IN:
            assert matrices.load(name).name == name

    def test_the_names_match_the_generated_data(self) -> None:
        assert set(matrices.BUILT_IN) == set(matrix_data.SCORES)
        assert set(matrices.BUILT_IN) == set(matrix_data.LABELS)
        assert set(matrices.BUILT_IN) == set(matrix_data.KINDS)
        assert set(matrices.BUILT_IN) == set(matrix_data.COMMENTS)

    def test_a_name_is_matched_whatever_its_case(self) -> None:
        assert matrices.load("blosum62").name == "BLOSUM62"
        assert matrices.load("nuc.4.4").name == "NUC.4.4"

    @pytest.mark.parametrize("name", matrices.BUILT_IN)
    def test_every_table_is_symmetric(self, name: str) -> None:
        assert matrices.load(name).symmetric

    @pytest.mark.parametrize("name", matrices.BUILT_IN)
    def test_every_table_is_square(self, name: str) -> None:
        matrix = matrices.load(name)
        assert len(matrix.scores) == len(matrix.labels)
        for row in matrix.scores:
            assert len(row) == len(matrix.labels)

    @pytest.mark.parametrize("name", matrices.BUILT_IN)
    def test_no_residue_is_listed_twice(self, name: str) -> None:
        labels = matrices.load(name).labels
        assert len(set(labels)) == len(labels)

    def test_blosum62_holds_its_published_values(self, blosum62) -> None:
        """Spot values from the NCBI's own BLOSUM62 file.

        Tryptophan against itself is the largest entry at 11 and the stop
        column holds the minimum, which is what the file's own header says.
        """
        assert blosum62.score("W", "W") == 11
        assert blosum62.score("A", "A") == 4
        assert blosum62.score("L", "I") == 2
        assert blosum62.score("W", "C") == -2
        assert blosum62.score("*", "*") == 1
        assert blosum62.lowest == -4
        assert blosum62.highest == 11

    def test_nuc44_holds_its_published_values(self, nuc44) -> None:
        assert nuc44.score("A", "A") == 5
        assert nuc44.score("A", "T") == -4
        assert nuc44.score("A", "N") == -2
        assert nuc44.score("A", "W") == 1
        assert nuc44.lowest == -4
        assert nuc44.highest == 5

    def test_an_ambiguity_code_may_score_below_zero_against_itself(
        self, nuc44
    ) -> None:
        """S stands for C or G, so two S residues agree only half the time.

        This is the reason identity and similarity are counted separately: here
        is an identical pair that is not a similar one.
        """
        assert nuc44.score("S", "S") == -1
        assert nuc44.score("N", "N") == -1

    def test_the_comment_lines_travel_with_the_table(self, blosum62) -> None:
        joined = " ".join(blosum62.comments)
        assert "1/2 Bit Units" in joined

    def test_the_pam_and_blosum_families_disagree(self) -> None:
        """A matrix is not interchangeable with another, which is the point.

        If these agreed everywhere, naming one rather than another would be
        decoration.
        """
        blosum = matrices.load("BLOSUM62")
        pam = matrices.load("PAM250")
        assert blosum.score("W", "W") != pam.score("W", "W")

    def test_nuc42_is_not_carried(self) -> None:
        assert "NUC.4.2" not in matrices.BUILT_IN


class TestSimple:
    def test_identical_residues_score_the_match(self, simple_dna) -> None:
        assert simple_dna.score("A", "A") == 1.0
        assert simple_dna.score("A", "C") == -1.0

    def test_an_ambiguity_code_is_an_ordinary_letter(self, simple_dna) -> None:
        assert simple_dna.score("N", "N") == 1.0
        assert simple_dna.score("N", "A") == -1.0

    def test_the_nucleotide_scheme_covers_t_and_u(self) -> None:
        matrix = matrices.simple(1.0, -1.0, "nucleotide")
        assert matrix.covers("T")
        assert matrix.covers("U")

    def test_the_protein_scheme_covers_the_stop(self) -> None:
        matrix = matrices.simple(1.0, -1.0, "protein")
        assert matrix.covers("*")
        assert matrix.covers("X")

    def test_a_scheme_is_symmetric(self) -> None:
        assert matrices.simple(2.0, -3.0, "protein").symmetric

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(MatrixError, match="unknown kind"):
            matrices.simple(1.0, -1.0, "rna")

    def test_the_comments_say_what_the_scheme_is(self) -> None:
        matrix = matrices.simple(3.0, -2.0, "nucleotide")
        joined = " ".join(matrix.comments)
        assert "3 for identical" in joined
        assert "-2 for anything else" in joined

    def test_fractional_scores_are_written_as_given(self) -> None:
        matrix = matrices.simple(1.5, -0.5, "nucleotide")
        assert "1.5 for identical" in " ".join(matrix.comments)


class TestParsing:
    TEXT = """# a tiny matrix
#
   A  C  G  T
A  2 -1 -1 -1
C -1  2 -1 -1
G -1 -1  2 -1
T -1 -1 -1  2
"""

    def test_a_matrix_file_is_read(self) -> None:
        matrix = matrices.parse(self.TEXT, "tiny", "tiny.txt")
        assert matrix.labels == "ACGT"
        assert matrix.score("A", "A") == 2
        assert matrix.kind == "nucleotide"

    def test_the_comments_are_kept(self) -> None:
        matrix = matrices.parse(self.TEXT, "tiny", "tiny.txt")
        assert matrix.comments[0] == "a tiny matrix"

    def test_protein_labels_make_a_protein_matrix(self) -> None:
        text = "   A  L\nA  2 -1\nL -1  2\n"
        assert matrices.parse(text, "x", "x").kind == "protein"

    def test_fractional_scores_are_accepted(self) -> None:
        text = "   A  C\nA  1.5 -0.5\nC -0.5  1.5\n"
        matrix = matrices.parse(text, "x", "x")
        assert matrix.score("A", "A") == 1.5

    def test_an_asymmetric_matrix_is_read_and_reported_as_such(self) -> None:
        text = "   A  C\nA  2 -1\nC -3  2\n"
        matrix = matrices.parse(text, "x", "x")
        assert not matrix.symmetric
        assert matrix.score("A", "C") == -1
        assert matrix.score("C", "A") == -3

    def test_a_non_square_table_is_refused(self) -> None:
        text = "   A  C\nA  2 -1\nC -1  2\nG -1 -1\n"
        with pytest.raises(MatrixError, match="has to be square"):
            matrices.parse(text, "x", "x")

    def test_a_short_row_is_refused(self) -> None:
        text = "   A  C\nA  2\nC -1  2\n"
        with pytest.raises(MatrixError, match="scores for"):
            matrices.parse(text, "x", "x")

    def test_a_repeated_column_is_refused(self) -> None:
        """Two columns of one name would silently keep the last.

        A matrix whose header says 'A C A' has an ambiguous second A, and
        reading it would score one residue by a column the file never meant.
        """
        text = "   A  C  A\nA  2 -1  2\nC -1  2 -1\nA  2 -1  2\n"
        with pytest.raises(MatrixError, match="twice"):
            matrices.parse(text, "x", "x")

    def test_a_score_that_is_not_a_number_is_refused(self) -> None:
        text = "   A  C\nA  2 nope\nC -1  2\n"
        with pytest.raises(MatrixError, match="not a number"):
            matrices.parse(text, "x", "x")

    @pytest.mark.parametrize("written", ["nan", "NaN", "inf", "-inf", "Infinity"])
    def test_a_score_that_is_not_finite_is_refused(self, written: str) -> None:
        """float() accepts 'nan' and 'inf', and neither is a score.

        Every comparison against NaN is false, so a matrix holding one
        produces an alignment with no columns and a score of NaN without any
        step reporting a problem - and NaN is not JSON either. The file is
        where it has to be caught.
        """
        text = f"   A  C\nA  {written} -1\nC -1  2\n"
        with pytest.raises(MatrixError, match="finite number"):
            matrices.parse(text, "x", "x")

    def test_the_offending_line_is_named(self) -> None:
        text = "   A  C\nA  2 -1\nC -1 nan\n"
        with pytest.raises(MatrixError, match="line 3"):
            matrices.parse(text, "x", "x")

    def test_an_empty_file_is_refused(self) -> None:
        with pytest.raises(MatrixError, match="no score table"):
            matrices.parse("# only comments\n", "x", "x")

    def test_a_file_is_read_from_disk(self, tmp_path) -> None:
        path = tmp_path / "tiny.txt"
        path.write_text(self.TEXT, encoding="utf-8")
        matrix = matrices.read(path)
        assert matrix.name == "tiny.txt"
        assert matrix.origin == str(path)

    def test_a_missing_file_is_refused(self, tmp_path) -> None:
        with pytest.raises(MatrixError, match="cannot read matrix"):
            matrices.read(tmp_path / "absent.txt")

    def test_a_path_is_loaded_when_it_is_not_a_built_in_name(self, tmp_path) -> None:
        path = tmp_path / "tiny.txt"
        path.write_text(self.TEXT, encoding="utf-8")
        assert matrices.load(str(path)).labels == "ACGT"

    def test_a_built_in_name_wins_over_a_file_of_that_name(self, tmp_path, monkeypatch) -> None:
        """A file called BLOSUM62 must not shadow the published table.

        Otherwise a score would depend on the working directory, which is the
        kind of difference nobody looks for.
        """
        (tmp_path / "BLOSUM62").write_text(self.TEXT, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert matrices.load("BLOSUM62").labels != "ACGT"
        assert matrices.load("./BLOSUM62").labels == "ACGT"

    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(MatrixError, match="BLOSUM62"):
            matrices.load("NOTAMATRIX")


class TestLookup:
    def test_a_residue_outside_the_table_is_named(self, blosum62) -> None:
        with pytest.raises(MatrixError, match="'J'"):
            blosum62.score("A", "J")

    def test_columns_names_the_offending_position(self, blosum62) -> None:
        with pytest.raises(MatrixError, match="position 3"):
            blosum62.columns("ACJ")

    def test_lookup_ignores_case(self, blosum62) -> None:
        assert blosum62.score("a", "A") == blosum62.score("A", "A")

    def test_unscored_lists_what_is_missing(self, blosum62) -> None:
        assert blosum62.unscored("ACDJUO") == ["J", "O", "U"]

    def test_a_row_is_the_table_row(self, blosum62) -> None:
        row = blosum62.row("A")
        assert row[blosum62.index["W"]] == blosum62.score("A", "W")

    def test_a_malformed_matrix_is_refused_at_construction(self) -> None:
        with pytest.raises(MatrixError, match="rows for"):
            Matrix("x", "protein", "AC", ((1, 2),))
        with pytest.raises(MatrixError, match="scores for"):
            Matrix("x", "protein", "AC", ((1, 2), (2,)))

    def test_a_non_finite_score_is_refused_at_construction(self) -> None:
        """An API caller can build a matrix without going through a file."""
        with pytest.raises(MatrixError, match="finite number"):
            Matrix("x", "protein", "AC", ((float("nan"), 2), (2, 1)))
        with pytest.raises(MatrixError, match="A against C"):
            Matrix("x", "protein", "AC", ((1, float("inf")), (float("inf"), 1)))

    @pytest.mark.parametrize("name", matrices.BUILT_IN)
    def test_every_built_in_score_is_finite(self, name: str) -> None:
        for row in matrices.load(name).scores:
            for value in row:
                assert math.isfinite(value)


class TestFold:
    def test_uracil_is_scored_as_thymine_on_a_thymine_matrix(self, nuc44) -> None:
        folded, reason = matrices.fold("ACGU", nuc44)
        assert folded == "ACGT"
        assert "uracil" in reason
        assert "NUC.4.4" in reason

    def test_nothing_is_folded_when_the_matrix_carries_both(self, simple_dna) -> None:
        folded, reason = matrices.fold("ACGU", simple_dna)
        assert folded == "ACGU"
        assert reason == ""

    def test_nothing_is_folded_for_a_protein_matrix(self, blosum62) -> None:
        folded, reason = matrices.fold("ACDU", blosum62)
        assert folded == "ACDU"
        assert reason == ""

    def test_one_residue_is_counted_in_the_singular(self, nuc44) -> None:
        _folded, reason = matrices.fold("ACGUA", nuc44)
        assert "1 uracil residue " in reason


class TestDescription:
    def test_describe_names_every_field_a_reader_needs(self, blosum62) -> None:
        described = dict(matrices.describe(blosum62))
        assert described["matrix"] == "BLOSUM62"
        assert described["scores"] == "protein"
        assert described["symmetric"] == "yes"
        assert described["origin"] == "built in"

    def test_the_grid_is_the_table(self, nuc44) -> None:
        columns, rows = matrices.grid(nuc44)
        assert columns == ["", *nuc44.labels]
        assert len(rows) == len(nuc44.labels)
        assert rows[0][0] == nuc44.labels[0]
        assert rows[0][1:] == list(nuc44.scores[0])

    def test_the_listing_names_every_built_in(self) -> None:
        names = [name for name, _kind, _summary in matrices.listing()]
        assert names == list(matrices.BUILT_IN)

    def test_every_listed_matrix_has_a_summary(self) -> None:
        for _name, _kind, summary in matrices.listing():
            assert summary.strip()


class TestUnscorable:
    def test_a_gapped_sequence_is_reported_as_already_aligned(self, blosum62) -> None:
        problems = matrices.unscorable({"query": "AC-DE"}, blosum62)
        assert len(problems) == 1
        assert "already aligned" in problems[0]
        assert "--degap" in problems[0]

    def test_an_unknown_residue_names_the_matrix(self, blosum62) -> None:
        problems = matrices.unscorable({"query": "ACDU"}, blosum62)
        assert "BLOSUM62 does not score 'U'" in problems[0]

    def test_a_scorable_sequence_has_no_problems(self, blosum62) -> None:
        assert matrices.unscorable({"query": "ACDEFG"}, blosum62) == []

    def test_a_gap_is_reported_before_an_unknown_residue(self, blosum62) -> None:
        """The two have different remedies, so the message must pick one.

        A sequence with both is already aligned, and removing the gaps is what
        the reader has to do first.
        """
        problems = matrices.unscorable({"query": "AC-U"}, blosum62)
        assert "already aligned" in problems[0]


class TestDefaults:
    def test_protein_gets_blosum62(self) -> None:
        assert matrices.default_for(PROTEIN).name == matrices.DEFAULT_PROTEIN

    def test_dna_and_rna_get_the_nucleotide_table(self) -> None:
        assert matrices.default_for(DNA).name == matrices.DEFAULT_NUCLEOTIDE
        assert matrices.default_for(RNA).name == matrices.DEFAULT_NUCLEOTIDE
