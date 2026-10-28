"""Putting a run together: pairing, matrix choice, strand, and what is said."""

from __future__ import annotations

import pytest

from align_pair import analysis
from align_pair.aligner import AlignmentError, Gaps
from align_pair.analysis import (
    Settings,
    analyse,
    choose_matrix,
    for_scoring,
    pair_up,
    prepare,
)
from align_pair.alphabet import resolve
from align_pair.matrices import MatrixError
from align_pair.reader import Record, SequenceError


def record(identifier: str, sequence: str) -> Record:
    return Record(identifier, "", sequence)


DNA_A = record("one", "ACGTACGTACGTACGTACGTACGT")
DNA_B = record("two", "ACGTACGTACGTACGTACGTACGA")
PEPTIDE_A = record("p1", "MVHLTPEEKSAVTALWGKVNVDEV")
PEPTIDE_B = record("p2", "MVHLTDAEKAAVSGLWGKVNADEV")


class TestSettings:
    def test_the_defaults_are_the_emboss_ones(self) -> None:
        settings = Settings()
        assert settings.mode == "global"
        assert settings.gaps == Gaps(10.0, 0.5)
        assert settings.matrix is None
        assert settings.strand == "plus"

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(AlignmentError, match="unknown alignment mode"):
            Settings(mode="semiglobal")

    def test_an_unknown_strand_is_refused(self) -> None:
        with pytest.raises(AlignmentError, match="unknown strand"):
            Settings(strand="forward")

    def test_an_unknown_pairing_is_refused(self) -> None:
        with pytest.raises(AlignmentError, match="unknown pairing"):
            Settings(pairing="diagonal")


class TestPairing:
    def test_one_input_pairs_every_record_with_every_later_one(self) -> None:
        records = [record("a", "AC"), record("b", "AC"), record("c", "AC")]
        pairs = pair_up(records, None)
        assert [(one.identifier, two.identifier) for one, two in pairs] == [
            ("a", "b"),
            ("a", "c"),
            ("b", "c"),
        ]

    def test_one_input_with_one_record_has_nothing_to_align(self) -> None:
        with pytest.raises(SequenceError, match="nothing to align it to"):
            pair_up([record("a", "AC")], None)

    def test_two_inputs_are_crossed_by_default(self) -> None:
        pairs = pair_up(
            [record("a", "AC"), record("b", "AC")], [record("x", "AC")], "all"
        )
        assert len(pairs) == 2

    def test_positional_pairing_takes_them_in_step(self) -> None:
        pairs = pair_up(
            [record("a", "AC"), record("b", "AC")],
            [record("x", "AC"), record("y", "AC")],
            "positional",
        )
        assert [(one.identifier, two.identifier) for one, two in pairs] == [
            ("a", "x"),
            ("b", "y"),
        ]

    def test_positional_pairing_refuses_unequal_counts(self) -> None:
        with pytest.raises(SequenceError, match="same number of records"):
            pair_up(
                [record("a", "AC"), record("b", "AC")], [record("x", "AC")], "positional"
            )


class TestMatrixChoice:
    def test_protein_gets_blosum62_by_default(self) -> None:
        detection = resolve("auto", PEPTIDE_A.sequence)
        matrix, notes = choose_matrix(Settings(), detection)
        assert matrix.name == "BLOSUM62"
        assert notes == []

    def test_nucleic_acid_gets_nuc44_by_default(self) -> None:
        detection = resolve("auto", DNA_A.sequence)
        matrix, _notes = choose_matrix(Settings(), detection)
        assert matrix.name == "NUC.4.4"

    def test_simple_builds_a_match_mismatch_scheme(self) -> None:
        detection = resolve("auto", DNA_A.sequence)
        matrix, _notes = choose_matrix(
            Settings(matrix="simple", match=3.0, mismatch=-2.0), detection
        )
        assert matrix.name == "simple"
        assert matrix.score("A", "A") == 3.0

    def test_a_protein_matrix_on_nucleic_acid_is_refused(self) -> None:
        """Scoring DNA with BLOSUM62 produces a plausible-looking number.

        Every nucleotide letter is also an amino-acid letter, so nothing would
        fail; the matrix simply knows nothing about transitions.
        """
        detection = resolve("auto", DNA_A.sequence)
        with pytest.raises(MatrixError, match="scores protein residues"):
            choose_matrix(Settings(matrix="BLOSUM62"), detection)

    def test_a_nucleotide_matrix_on_protein_is_refused(self) -> None:
        detection = resolve("auto", PEPTIDE_A.sequence)
        with pytest.raises(MatrixError, match="scores nucleotide residues"):
            choose_matrix(Settings(matrix="NUC.4.4"), detection)

    def test_an_asymmetric_matrix_is_reported(self, tmp_path) -> None:
        path = tmp_path / "skew.txt"
        path.write_text("   A  C\nA  2 -1\nC -3  2\n", encoding="utf-8")
        detection = resolve("dna", "ACAC")
        _matrix, notes = choose_matrix(Settings(matrix=str(path)), detection)
        assert any("not symmetric" in note for note in notes)


class TestPrepare:
    def test_a_clean_sequence_passes_through(self, blosum62) -> None:
        sequence, notes = prepare(PEPTIDE_A, Settings(), blosum62)
        assert sequence == PEPTIDE_A.sequence
        assert notes == []

    def test_a_gapped_sequence_is_refused_without_degap(self, blosum62) -> None:
        gapped = record("g", "MVHL-TPEEK")
        with pytest.raises(MatrixError, match="already aligned"):
            prepare(gapped, Settings(), blosum62)

    def test_degap_removes_the_gaps_and_says_so(self, blosum62) -> None:
        gapped = record("g", "MVHL-TPEEK")
        sequence, notes = prepare(gapped, Settings(degap=True), blosum62)
        assert sequence == "MVHLTPEEK"
        assert any("1 gap character removed" in note for note in notes)

    def test_an_empty_record_is_refused(self, blosum62) -> None:
        with pytest.raises(SequenceError, match="no residues to align"):
            prepare(record("e", ""), Settings(), blosum62)

    def test_a_record_emptied_by_degap_says_which(self, blosum62) -> None:
        with pytest.raises(SequenceError, match="gap characters are removed"):
            prepare(record("e", "----"), Settings(degap=True), blosum62)

    def test_prepare_leaves_the_record_in_its_own_alphabet(self, nuc44) -> None:
        """Folding onto the matrix's letters is not prepare's job.

        It has to happen after the strand is chosen, or a reverse complement is
        taken over letters the folding put there.
        """
        sequence, notes = prepare(record("r", "ACGU"), Settings(), nuc44)
        assert sequence == "ACGU"
        assert notes == []

    def test_rna_is_folded_onto_the_thymine_matrix_for_scoring(self, nuc44) -> None:
        sequence, notes = for_scoring("ACGU", nuc44, "r")
        assert sequence == "ACGT"
        assert any("uracil" in note for note in notes)

    def test_a_residue_no_matrix_scores_is_refused(self, blosum62) -> None:
        with pytest.raises(MatrixError, match="does not score 'O'"):
            for_scoring("MVHLO", blosum62, "x")


class TestAnalyse:
    def test_a_pair_is_aligned_and_measured(self) -> None:
        run = analyse([PEPTIDE_A], [PEPTIDE_B], Settings())
        assert len(run.pairs) == 1
        pair = run.pairs[0]
        assert pair.metrics.length == 24
        assert pair.label == "p1 vs p2"

    def test_the_matrix_and_alphabet_are_carried_on_the_run(self) -> None:
        run = analyse([PEPTIDE_A], [PEPTIDE_B], Settings())
        assert run.matrix.name == "BLOSUM62"
        assert run.alphabet_label == "protein"

    def test_several_pairs_are_aligned(self) -> None:
        run = analyse([DNA_A, DNA_B], None, Settings())
        assert len(run.pairs) == 1
        run = analyse([DNA_A, DNA_B], [DNA_A, DNA_B], Settings())
        assert len(run.pairs) == 4

    def test_a_pair_too_large_is_refused_before_it_is_run(self) -> None:
        big = record("big", "A" * 500)
        with pytest.raises(AlignmentError, match="--max-cells"):
            analyse([big], [big], Settings(max_cells=1000))

    def test_the_refusal_names_the_pair_and_the_cost(self) -> None:
        big = record("big", "A" * 500)
        with pytest.raises(AlignmentError, match="251,001 cells"):
            analyse([big], [big], Settings(max_cells=1000))

    def test_sources_are_carried_for_the_output_to_name(self) -> None:
        run = analyse([PEPTIDE_A], [PEPTIDE_B], Settings(), ["one.fa", "two.fa"])
        assert run.sources == ("one.fa", "two.fa")


class TestStrand:
    REVERSE = record("rc", "TCGTACGTACGTACGTACGTACGT")

    def test_the_minus_strand_is_the_reverse_complement(self) -> None:
        run = analyse([DNA_A], [DNA_A], Settings(strand="minus"))
        pair = run.pairs[0]
        assert pair.strand == "minus"

    def test_both_strands_are_tried_and_the_better_kept(self) -> None:
        from align_pair.alphabet import DNA, reverse_complement

        # (ACGT) repeated is its own reverse complement, so the sequence used
        # here is deliberately not palindromic.
        forward = record("fwd", "ACGGGTTACGTTACGTACGTAAGT")
        flipped = record("flip", reverse_complement(forward.sequence, DNA))
        run = analyse([forward], [flipped], Settings(strand="both"))
        pair = run.pairs[0]
        assert pair.strand == "minus"
        assert pair.other_strand_score is not None
        assert pair.alignment.score > pair.other_strand_score
        assert any("both strands" in note for note in pair.qualifications)

    def test_the_plus_strand_is_kept_on_a_tie(self) -> None:
        """A palindrome scores the same either way round.

        Keeping the plus strand makes the reported strand a property of the
        sequences rather than of which comparison happened to run last.
        """
        palindrome = record("pal", "ACGTACGT")
        run = analyse([palindrome], [palindrome], Settings(strand="both"))
        assert run.pairs[0].strand == "plus"

    def test_a_tie_is_reported_as_a_tie(self) -> None:
        """Keeping plus on a tie is not plus winning.

        'the plus strand scored better, 40 against 40' says two things that
        cannot both be true, and a reader would take the first.
        """
        palindrome = record("pal", "ACGTACGT")
        run = analyse([palindrome], [palindrome], Settings(strand="both"))
        notes = [note for note in run.pairs[0].qualifications if "strand" in note]
        assert notes
        assert any("scored the same" in note for note in notes)
        assert not any("scored better" in note for note in notes)

    def test_a_real_difference_is_still_reported_as_one(self) -> None:
        from align_pair.alphabet import DNA, reverse_complement

        forward = record("fwd", "ACGGGTTACGTTACGTACGTAAGT")
        flipped = record("flip", reverse_complement(forward.sequence, DNA))
        run = analyse([forward], [flipped], Settings(strand="both"))
        notes = [note for note in run.pairs[0].qualifications if "strand" in note]
        assert any("scored better" in note for note in notes)

    def test_the_winning_score_is_quoted_first(self) -> None:
        """'better, X against Y' has to put the better score where X is."""
        from align_pair.alphabet import DNA, reverse_complement

        forward = record("fwd", "ACGGGTTACGTTACGTACGTAAGT")
        flipped = record("flip", reverse_complement(forward.sequence, DNA))
        run = analyse([forward], [flipped], Settings(strand="both"))
        pair = run.pairs[0]
        note = next(note for note in pair.qualifications if "scored better" in note)
        quoted = note.rsplit(",", 1)[1]
        better, worse = (float(part) for part in quoted.split("against"))
        assert better == pair.alignment.score
        assert worse == pair.other_strand_score
        assert better > worse

    def test_coordinates_are_mapped_back_onto_the_record(self) -> None:
        """A minus-strand interval is reported in the file's own numbering.

        A reader checking the region against the file needs positions in the
        file, not positions along a reverse complement the file does not hold.
        """
        from align_pair.alphabet import DNA, reverse_complement

        target = record("t", reverse_complement("ACGTACGTAC", DNA) + "TTTTT")
        run = analyse(
            [record("q", "ACGTACGTAC")], [target], Settings(strand="minus", mode="glocal")
        )
        pair = run.pairs[0]
        start, end = pair.span_second
        assert target.sequence[start - 1 : end] == reverse_complement(
            pair.alignment.aligned_second.replace("-", ""), DNA
        )

    @pytest.mark.parametrize("strand", ["minus", "both"])
    def test_rna_aligns_on_either_strand(self, strand: str) -> None:
        """The complement is taken before anything is folded onto the matrix.

        NUC.4.4 is indexed by T, so RNA has to be folded to be scored. Folding
        first and complementing after would leave a T the RNA complement table
        does not translate, and turn its A into a U the table cannot score.
        """
        rna = record("r", "ACGUACGUACGUACGUACGUACGU")
        run = analyse([rna], [rna], Settings(strand=strand))
        assert run.pairs[0].metrics.identity == 1.0

    def test_rna_finds_its_own_reverse_complement(self) -> None:
        from align_pair.alphabet import RNA, reverse_complement

        sequence = "ACGGGUUACGUUACGUACGUAAGU"
        flipped = record("rc", reverse_complement(sequence, RNA))
        run = analyse([record("r", sequence)], [flipped], Settings(strand="both"))
        pair = run.pairs[0]
        assert pair.strand == "minus"
        assert pair.metrics.identity == 1.0
        assert pair.metrics.gaps == 0

    def test_the_fold_is_reported_for_the_strand_that_was_kept(self) -> None:
        """Complementing changes how many uracils there are to fold."""
        from align_pair.alphabet import RNA, reverse_complement

        sequence = "ACGGGUUACGUUACGUACGUAAGU"
        flipped = record("rc", reverse_complement(sequence, RNA))
        run = analyse([record("r", sequence)], [flipped], Settings(strand="minus"))
        notes = [n for n in run.pairs[0].qualifications if "uracil" in n]
        assert len(notes) == 2

    def test_a_strand_is_refused_for_protein(self) -> None:
        with pytest.raises(AlignmentError, match="only meaningful for nucleic acid"):
            analyse([PEPTIDE_A], [PEPTIDE_B], Settings(strand="both"))


class TestMixedAlphabets:
    """One matrix scores every pair, so the batch has to be one molecule.

    Detecting over the records together lets one long protein carry a file of
    DNA into a protein matrix, where even the DNA-against-DNA pair is scored by
    a table that knows nothing about transitions - and nothing fails, because
    every nucleotide letter is also an amino-acid letter.
    """

    PROTEIN = record("prot", "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLS")

    def test_a_mixed_batch_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="not all the same kind"):
            analyse([DNA_A, DNA_B, self.PROTEIN], None, Settings())

    def test_the_refusal_names_the_odd_records(self) -> None:
        with pytest.raises(SequenceError, match="prot looks like protein"):
            analyse([DNA_A, DNA_B, self.PROTEIN], None, Settings())

    def test_a_mixed_pair_across_two_inputs_is_refused(self) -> None:
        with pytest.raises(SequenceError, match="not all the same kind"):
            analyse([DNA_A], [self.PROTEIN], Settings())

    def test_naming_the_alphabet_allows_it_and_says_so(self) -> None:
        """The override is the documented way through, not a way to hide it."""
        run = analyse(
            [DNA_A, DNA_B, self.PROTEIN], None, Settings(alphabet="protein")
        )
        assert run.matrix.name == "BLOSUM62"
        assert any(
            "not all the same kind" in note for note in run.qualifications
        )

    def test_a_clean_batch_is_not_refused(self) -> None:
        assert len(analyse([DNA_A, DNA_B], None, Settings()).pairs) == 1
        assert len(analyse([PEPTIDE_A, PEPTIDE_B], None, Settings()).pairs) == 1

    def test_dna_and_rna_together_are_not_mixed(self) -> None:
        """Both are nucleic acid, and one nucleotide matrix scores both."""
        rna = record("rna", "ACGUACGUACGUACGUACGUACGU")
        run = analyse([DNA_A, rna], None, Settings())
        assert run.matrix.name == "NUC.4.4"


class TestSizeGuard:
    def test_a_band_does_not_hide_the_memory_it_still_holds(self) -> None:
        """Cells measure time and the footprint measures memory.

        A band cuts the cells to its own width while the rows still span the
        second sequence, so a banded run can be quick and still large: a
        40,000-residue pair at band zero is 40,001 cells and about 19 MB.
        """
        from align_pair.aligner import cells as count, footprint

        assert count(40000, 40000, 0) == 40001
        held = footprint(40000, 40000, 0)
        assert 15_000_000 < held < 25_000_000
        big = record("big", "A" * 40000)
        with pytest.raises(AlignmentError, match="MB"):
            analyse([big], [big], Settings(band=0, max_memory=10_000_000))

    def test_the_cell_limit_still_applies(self) -> None:
        big = record("big", "A" * 500)
        with pytest.raises(AlignmentError, match="--max-cells"):
            analyse([big], [big], Settings(max_cells=1000))

    def test_a_memory_refusal_says_a_band_will_not_help(self) -> None:
        big = record("big", "A" * 40000)
        with pytest.raises(AlignmentError, match="A band does not help"):
            analyse([big], [big], Settings(band=0, max_memory=10_000_000))

    def test_the_footprint_grows_with_the_band(self) -> None:
        from align_pair.aligner import footprint

        narrow = footprint(10000, 10000, 10)
        wide = footprint(10000, 10000, 1000)
        assert narrow < wide < footprint(10000, 10000, None)


class TestQualifications:
    def test_free_gaps_are_reported(self) -> None:
        run = analyse([DNA_A], [DNA_B], Settings(gaps=Gaps(0.0, 0.0)))
        assert any("cost nothing" in note for note in run.qualifications)

    def test_an_extension_dearer_than_an_opening_is_reported(self) -> None:
        run = analyse([DNA_A], [DNA_B], Settings(gaps=Gaps(1.0, 3.0)))
        assert any("penalised more heavily" in note for note in run.qualifications)

    def test_a_weakly_supported_alphabet_is_reported(self) -> None:
        run = analyse([record("a", "ACGT")], [record("b", "ACGT")], Settings())
        assert any("read from only" in note for note in run.qualifications)

    def test_a_named_alphabet_against_the_evidence_is_reported(self) -> None:
        run = analyse([DNA_A], [DNA_B], Settings(alphabet="protein"))
        assert any(
            "because --alphabet said so" in note for note in run.qualifications
        )

    def test_co_optimal_alignments_are_reported(self) -> None:
        run = analyse(
            [record("a", "AAA")], [record("b", "AA")], Settings(matrix="simple")
        )
        pair = run.pairs[0]
        assert pair.alignment.co_optimal > 1
        assert any("share this score" in note for note in pair.qualifications)

    def test_a_unique_alignment_says_nothing_about_co_optimality(self) -> None:
        run = analyse([PEPTIDE_A], [PEPTIDE_B], Settings())
        pair = run.pairs[0]
        assert pair.alignment.co_optimal == 1
        assert not any("share this score" in note for note in pair.qualifications)

    def test_a_band_reports_that_the_score_is_a_lower_bound(self) -> None:
        run = analyse([DNA_A], [DNA_B], Settings(band=2))
        assert any("lower bound" in note for note in run.pairs[0].qualifications)

    def test_an_identical_pair_the_matrix_dislikes_is_reported(self) -> None:
        # One ambiguity code, not four: an ambiguity code is not a nucleotide
        # code, so a sequence with too many of them is read as protein and
        # scored by BLOSUM62, where S is serine and S against S is 4.
        run = analyse(
            [record("a", "ACGTSACGTACGTACGTACGTACG")],
            [record("b", "ACGTSACGTACGTACGTACGTACG")],
            Settings(),
        )
        notes = run.pairs[0].qualifications
        assert any("count towards identity but not" in note for note in notes)
        assert any("ambiguity code with itself" in note for note in notes)

    def test_an_empty_local_alignment_is_reported(self) -> None:
        run = analyse(
            [record("a", "AAAAAAAAAAAAAAAAAAAAAAAA")],
            [record("b", "CCCCCCCCCCCCCCCCCCCCCCCC")],
            Settings(mode="local"),
        )
        pair = run.pairs[0]
        assert pair.alignment.empty
        assert any("empty one" in note for note in pair.qualifications)


class TestConstants:
    def test_the_pairings_and_strands_are_listed(self) -> None:
        assert analysis.PAIRINGS == ("all", "positional")
        assert analysis.STRANDS == ("plus", "minus", "both")

    def test_the_cell_limit_is_a_round_number(self) -> None:
        assert analysis.DEFAULT_MAX_CELLS == 100_000_000
