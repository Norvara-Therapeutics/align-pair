"""Deciding what a sequence is written in, and the alphabets themselves."""

from __future__ import annotations

import pytest

from align_pair import alphabet
from align_pair.alphabet import DNA, PROTEIN, RNA


class TestAlphabets:
    def test_every_alphabet_is_reachable_by_name(self) -> None:
        for name in ("dna", "rna", "protein"):
            assert alphabet.get(name).name == name

    def test_a_name_is_matched_whatever_its_case(self) -> None:
        assert alphabet.get("DNA") is DNA

    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="protein"):
            alphabet.get("peptide")

    def test_nucleotide_alphabets_know_they_are_nucleotide(self) -> None:
        assert DNA.is_nucleotide
        assert RNA.is_nucleotide
        assert not PROTEIN.is_nucleotide

    def test_lengths_are_written_in_the_alphabet_unit(self) -> None:
        assert DNA.unit == "bp"
        assert RNA.unit == "bp"
        assert PROTEIN.unit == "aa"

    def test_a_nucleotide_alphabet_does_not_accept_a_stop(self) -> None:
        assert "*" not in DNA.valid
        assert "*" in PROTEIN.valid

    def test_gaps_are_accepted_by_every_alphabet(self) -> None:
        for each in (DNA, RNA, PROTEIN):
            assert alphabet.GAPS <= each.valid

    def test_the_rare_residues_are_valid_protein(self) -> None:
        assert "U" in PROTEIN.valid
        assert "O" in PROTEIN.valid
        assert PROTEIN.rare["U"] == "selenocysteine"

    def test_ambiguity_codes_are_recognised(self) -> None:
        assert DNA.is_ambiguous("N")
        assert DNA.is_ambiguous("s")
        assert not DNA.is_ambiguous("A")
        assert PROTEIN.is_ambiguous("X")
        assert not PROTEIN.is_ambiguous("A")

    def test_the_rna_codes_stand_for_uracil(self) -> None:
        assert RNA.ambiguity["Y"] == "CU"
        assert DNA.ambiguity["Y"] == "CT"


class TestDetection:
    def test_plain_dna_is_dna(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGTACGTACGTACGTACGTACGT"))
        assert detected.alphabet is DNA
        assert detected.certain

    def test_more_uracil_than_thymine_is_rna(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGUACGUACGUACGUACGUACGU"))
        assert detected.alphabet is RNA

    def test_a_file_mixing_the_two_is_read_as_dna(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGTACGTACGTACGUACGTACGT"))
        assert detected.alphabet is DNA

    def test_a_peptide_is_protein(self) -> None:
        detected = alphabet.detect(alphabet.census("MVHLTPEEKSAVTALWGKVNVDEV"))
        assert detected.alphabet is PROTEIN
        assert detected.certain

    def test_a_short_sequence_is_reported_as_weakly_supported(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGT"))
        assert not detected.certain
        assert "only 4 residues" in detected.qualification

    def test_nothing_to_judge_from_says_so(self) -> None:
        detected = alphabet.detect({})
        assert detected.letters == 0
        assert "no residues to judge from" in detected.qualification

    def test_a_certain_detection_has_nothing_to_qualify(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGTACGTACGTACGTACGTACGT"))
        assert detected.qualification == ""

    def test_gaps_and_stops_are_not_counted(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGT----ACGT----ACGTACGT"))
        assert detected.letters == 16

    def test_no_letter_settles_the_alphabet_on_its_own(self) -> None:
        """Every nucleotide code is also an amino-acid code.

        T is threonine, N asparagine, R arginine. Detection is by share, which
        is why a protein rich in those residues needs its alphabet naming.
        """
        for residue in "ACGTNRYSWKMBDHV":
            assert residue in PROTEIN.valid

    def test_the_share_is_reported_whatever_the_decision(self) -> None:
        detected = alphabet.detect(alphabet.census("ACGTACGTACGTACGTACGTACGM"))
        assert 0.9 <= detected.nucleotide_share < 1.0


class TestResolve:
    def test_auto_detects(self) -> None:
        detected = alphabet.resolve("auto", "ACGTACGTACGTACGTACGTACGT")
        assert detected.alphabet is DNA
        assert not detected.requested

    def test_a_named_alphabet_is_taken_as_given(self) -> None:
        detected = alphabet.resolve("protein", "ACGTACGTACGTACGTACGTACGT")
        assert detected.alphabet is PROTEIN
        assert detected.requested
        assert detected.certain

    def test_the_share_is_still_measured_when_the_alphabet_is_named(self) -> None:
        """Naming the alphabet does not hide what the sequence looks like.

        A sequence named as protein that is 100% nucleotide codes is probably
        a mistake, and the output can only say so if the share was measured.
        """
        detected = alphabet.resolve("protein", "ACGTACGTACGTACGTACGTACGT")
        assert detected.nucleotide_share == 1.0

    def test_several_sequences_are_judged_together(self) -> None:
        detected = alphabet.resolve(
            "auto", "ACGTACGTACGTACGT", "MVHLTPEEKSAVTALWGKV"
        )
        assert detected.alphabet is PROTEIN

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(KeyError):
            alphabet.resolve("peptide", "ACGT")


class TestReverseComplement:
    def test_dna_is_complemented_and_reversed(self) -> None:
        assert alphabet.reverse_complement("ACGT", DNA) == "ACGT"
        assert alphabet.reverse_complement("AAAC", DNA) == "GTTT"

    def test_rna_uses_uracil(self) -> None:
        assert alphabet.reverse_complement("ACGU", RNA) == "ACGU"
        assert alphabet.reverse_complement("AAAC", RNA) == "GUUU"

    def test_ambiguity_codes_are_complemented(self) -> None:
        assert alphabet.reverse_complement("RYSWKM", DNA) == "KMWSRY"
        assert alphabet.reverse_complement("BDHV", DNA) == "BDHV"
        assert alphabet.reverse_complement("N", DNA) == "N"

    def test_the_length_is_unchanged(self) -> None:
        assert len(alphabet.reverse_complement("ACGT-NX", DNA)) == 7

    def test_two_passes_return_the_original(self) -> None:
        sequence = "ACGTTGCAnnRY"
        once = alphabet.reverse_complement(sequence, DNA)
        assert alphabet.reverse_complement(once, DNA) == sequence

    def test_protein_has_no_complement(self) -> None:
        with pytest.raises(ValueError, match="no complement"):
            alphabet.reverse_complement("ACDE", PROTEIN)


class TestGaps:
    def test_every_gap_character_is_removed(self) -> None:
        assert alphabet.degapped("AC-G.T~A") == "ACGTA"

    def test_a_sequence_without_gaps_is_unchanged(self) -> None:
        assert alphabet.degapped("ACGT") == "ACGT"

    def test_gaps_are_counted(self) -> None:
        assert alphabet.gap_count("AC-G.T~A") == 3
        assert alphabet.gap_count("ACGT") == 0

    def test_the_written_gap_is_a_hyphen(self) -> None:
        assert alphabet.GAP == "-"
        assert alphabet.GAP in alphabet.GAPS


class TestUnknownResidues:
    def test_a_residue_outside_the_alphabet_is_listed(self) -> None:
        counts = alphabet.census("ACGTZ")
        assert alphabet.unknown_residues(counts, DNA) == ["Z"]

    def test_a_clean_sequence_lists_nothing(self) -> None:
        counts = alphabet.census("ACGTN-")
        assert alphabet.unknown_residues(counts, DNA) == []

    def test_uracil_is_unknown_to_dna(self) -> None:
        counts = alphabet.census("ACGU")
        assert alphabet.unknown_residues(counts, DNA) == ["U"]


class TestCensus:
    def test_residues_are_counted_case_insensitively(self) -> None:
        assert alphabet.census("AaCc") == {"A": 2, "C": 2}

    def test_several_sequences_are_added_up(self) -> None:
        assert alphabet.census("AA", "AC") == {"A": 3, "C": 1}

    def test_an_empty_input_counts_nothing(self) -> None:
        assert alphabet.census("") == {}
