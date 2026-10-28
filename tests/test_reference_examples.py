"""The example records, against facts taken from outside this package.

Every expectation here is either a published property of the molecule or
something computable from the records without aligning anything, so these tests
say the alignments are right about real sequence rather than consistent with a
previous run.
"""

from __future__ import annotations

import pytest

from align_pair import matrices, reader
from align_pair.aligner import Gaps, align
from align_pair.analysis import Settings, analyse


def records(examples, name):
    return reader.read_all(examples / name)


def sequence(examples, name) -> str:
    return records(examples, name)[0].sequence


class TestTheRecordsThemselves:
    def test_the_globins_are_the_expected_records(self, examples) -> None:
        human = records(examples, "hbb_human.fasta")[0]
        mouse = records(examples, "hbb_mouse.fasta")[0]
        assert human.identifier == "NP_000509.1"
        assert mouse.identifier == "NP_032246.2"
        assert len(human) == 147
        assert len(mouse) == 147

    def test_both_globins_carry_the_initiator_methionine(self, examples) -> None:
        """The RefSeq protein records include the initiator Met.

        The mature beta chain is 146 residues, so every position quoted from
        the literature sits one later in these records.
        """
        assert sequence(examples, "hbb_human.fasta").startswith("MVHLT")
        assert sequence(examples, "hbb_mouse.fasta").startswith("MVHLT")

    def test_the_haem_binding_histidines_are_conserved(self, examples) -> None:
        """The distal (E7) and proximal (F8) histidines of beta globin.

        His63 and His92 of the mature chain hold the haem iron; with the
        initiator methionine they are positions 64 and 93 here. They are
        invariant across vertebrate beta globins, and an alignment that failed
        to pair them would be wrong whatever it scored.
        """
        for name in ("hbb_human.fasta", "hbb_mouse.fasta"):
            residues = sequence(examples, name)
            assert residues[63] == "H"
            assert residues[92] == "H"

    def test_the_capsid_proteins_are_the_expected_records(self, examples) -> None:
        aav2 = records(examples, "aav2_vp1.fasta")[0]
        aav9 = records(examples, "aav9_vp1.fasta")[0]
        assert aav2.identifier == "YP_680426.1"
        assert aav9.identifier == "AAS99264.1"
        assert len(aav2) == 735
        assert len(aav9) == 736

    def test_both_capsids_carry_the_phospholipase_motif(self, examples) -> None:
        """The HDxxY catalytic motif of the VP1u phospholipase A2 domain.

        It sits at residues 75 to 79 of VP1 in both serotypes and is required
        for the virus to escape the endosome.
        """
        assert sequence(examples, "aav2_vp1.fasta")[74:79] == "HDKAY"
        assert sequence(examples, "aav9_vp1.fasta")[74:79] == "HDKAY"

    def test_the_globin_mrnas_are_longer_than_their_coding_sequence(
        self, examples
    ) -> None:
        """Each mRNA carries untranslated regions around a 441 nt coding part.

        A 147-residue protein needs 441 nucleotides plus a stop, so most of the
        difference between 628 and 646 bases is untranslated - which is what
        makes this pair worth aligning locally as well as globally.
        """
        assert len(sequence(examples, "hbb_human_mrna.fasta")) == 628
        assert len(sequence(examples, "hbb_mouse_mrna.fasta")) == 646
        assert 3 * 147 + 3 == 444


@pytest.fixture(scope="session")
def globin_run(examples):
    return analyse(
        records(examples, "hbb_human.fasta"),
        records(examples, "hbb_mouse.fasta"),
        Settings(),
    )


@pytest.fixture(scope="session")
def capsid_run(examples):
    return analyse(
        records(examples, "aav2_vp1.fasta"),
        records(examples, "aav9_vp1.fasta"),
        Settings(),
    )


class TestGlobinAlignment:
    def test_the_optimal_alignment_is_gapless(self, globin_run) -> None:
        """Two equal-length homologues this close need no gaps.

        Opening one costs ten and would have to buy back more than ten in
        substitution score; nothing in this pair does.
        """
        pair = globin_run.pairs[0]
        assert pair.metrics.gaps == 0
        assert pair.metrics.length == 147

    def test_the_identity_is_the_positional_identity(self, examples, globin_run) -> None:
        """A gapless alignment of equal-length sequences is the obvious one.

        Its identity count can therefore be checked without aligning anything,
        which is an independent arithmetic on the same records.
        """
        human = sequence(examples, "hbb_human.fasta")
        mouse = sequence(examples, "hbb_mouse.fasta")
        expected = sum(one == two for one, two in zip(human, mouse))
        assert expected == 118
        assert globin_run.pairs[0].metrics.identities == expected

    def test_the_score_is_the_sum_of_its_columns(self, globin_run) -> None:
        pair = globin_run.pairs[0]
        matrix = globin_run.matrix
        total = sum(
            matrix.score(one, two)
            for one, two in zip(pair.alignment.aligned_first, pair.alignment.aligned_second)
        )
        assert total == pair.alignment.score

    def test_the_haem_histidines_are_aligned_to_each_other(self, globin_run) -> None:
        pair = globin_run.pairs[0]
        for column in (63, 92):
            assert pair.alignment.aligned_first[column] == "H"
            assert pair.alignment.aligned_second[column] == "H"

    def test_the_alignment_is_the_only_optimal_one(self, globin_run) -> None:
        assert globin_run.pairs[0].alignment.co_optimal == 1

    def test_every_mode_agrees_on_this_pair(self, examples) -> None:
        """Two sequences of one length aligning end to end leave no overhang.

        With nothing to trim, the four modes have nothing to disagree about,
        and a mode that reported something different here would be wrong.
        """
        scores = set()
        for mode in ("global", "glocal", "overlap", "local"):
            run = analyse(
                records(examples, "hbb_human.fasta"),
                records(examples, "hbb_mouse.fasta"),
                Settings(mode=mode),
            )
            scores.add(run.pairs[0].alignment.score)
        assert len(scores) == 1


class TestCapsidAlignment:
    def test_the_serotypes_are_about_four_fifths_identical(self, capsid_run) -> None:
        """Published pairwise identity for AAV2 and AAV9 VP1 is near 82%.

        It is quoted here as a range rather than a figure, since the exact
        number depends on the alignment and that is what is being tested.
        """
        identity = capsid_run.pairs[0].metrics.identity
        assert 0.80 < identity < 0.84

    def test_the_length_difference_forces_at_least_one_gap(self, capsid_run) -> None:
        pair = capsid_run.pairs[0]
        assert pair.metrics.length_first != pair.metrics.length_second
        assert pair.metrics.gaps >= 1

    def test_the_phospholipase_motif_is_aligned_to_itself(self, capsid_run) -> None:
        """Both carry HDKAY at 75-79, so a correct alignment pairs them.

        The motif is upstream of the first gap, so its columns are its
        positions.
        """
        pair = capsid_run.pairs[0]
        assert pair.alignment.aligned_first[74:79] == "HDKAY"
        assert pair.alignment.aligned_second[74:79] == "HDKAY"

    def test_both_capsids_start_at_the_same_residue(self, capsid_run) -> None:
        pair = capsid_run.pairs[0]
        assert pair.alignment.aligned_first[:12] == "MAADGYLPDWLE"
        assert pair.alignment.aligned_second[:12] == "MAADGYLPDWLE"

    def test_a_band_of_twenty_finds_the_same_score(self, examples, capsid_run) -> None:
        """The two differ by one residue, so the path stays near the diagonal.

        A band wide enough to hold it gives the unbanded answer at a fraction
        of the work, which is the case bands are for.
        """
        banded = analyse(
            records(examples, "aav2_vp1.fasta"),
            records(examples, "aav9_vp1.fasta"),
            Settings(band=20),
        )
        assert banded.pairs[0].alignment.score == capsid_run.pairs[0].alignment.score
        assert banded.pairs[0].alignment.cells < capsid_run.pairs[0].alignment.cells / 10


class TestMessengerRnaAlignment:
    def test_the_untranslated_ends_are_what_local_mode_trims(
        self, examples
    ) -> None:
        """Global mode must account for the ends; local mode need not.

        The two mRNAs differ in length by 18 bases and diverge at their ends,
        so the local score is the higher of the two and the local alignment the
        shorter.
        """
        first = records(examples, "hbb_human_mrna.fasta")
        second = records(examples, "hbb_mouse_mrna.fasta")
        globally = analyse(first, second, Settings(mode="global")).pairs[0]
        locally = analyse(first, second, Settings(mode="local")).pairs[0]
        assert locally.alignment.score > globally.alignment.score
        assert locally.metrics.length < globally.metrics.length

    def test_the_alphabet_is_detected_as_dna(self, examples) -> None:
        run = analyse(
            records(examples, "hbb_human_mrna.fasta"),
            records(examples, "hbb_mouse_mrna.fasta"),
            Settings(),
        )
        assert run.detection.alphabet.name == "dna"
        assert run.matrix.name == "NUC.4.4"

    def test_an_mrna_aligns_to_its_own_reverse_complement_on_the_minus_strand(
        self, examples
    ) -> None:
        """A sequence is perfectly identical to the reverse complement of its
        reverse complement, which is the one case where the answer is known
        exactly.
        """
        from align_pair.alphabet import DNA, reverse_complement

        human = records(examples, "hbb_human_mrna.fasta")
        flipped = [
            reader.Record("flipped", "", reverse_complement(human[0].sequence, DNA))
        ]
        run = analyse(human, flipped, Settings(strand="both"))
        pair = run.pairs[0]
        assert pair.strand == "minus"
        assert pair.metrics.identity == 1.0
        assert pair.metrics.gaps == 0


class TestAgainstTheMatrixItself:
    def test_an_identical_protein_scores_its_own_diagonal(self, examples) -> None:
        """Aligning a sequence with itself must score the sum of its diagonal.

        Nothing about the recurrence can change that, so it is the cheapest
        check there is that the matrix is being indexed correctly.
        """
        residues = sequence(examples, "aav2_vp1.fasta")
        matrix = matrices.load("BLOSUM62")
        alignment = align(residues, residues, matrix, Gaps(10.0, 0.5))
        expected = sum(matrix.score(residue, residue) for residue in residues)
        assert alignment.score == expected
        assert alignment.co_optimal == 1
        assert alignment.aligned_first == residues
