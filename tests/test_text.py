"""Number formatting for human-facing output."""

from __future__ import annotations

from align_pair import text
from align_pair.alphabet import DNA, PROTEIN


class TestPercent:
    def test_a_share_is_written_to_one_place(self) -> None:
        assert text.percent(80.27) == "80.3%"

    def test_a_near_miss_is_never_rounded_to_a_hundred(self) -> None:
        """An alignment with one column differing is not 100% identical.

        A reader will repeat '100% identical'; the bound says what is true.
        """
        assert text.percent(99.98) == ">99.9%"

    def test_a_near_miss_is_never_rounded_to_zero(self) -> None:
        assert text.percent(0.004) == "<0.1%"

    def test_exactly_a_hundred_is_written_plainly(self) -> None:
        assert text.percent(100.0) == "100.0%"

    def test_exactly_zero_is_written_plainly(self) -> None:
        assert text.percent(0.0) == "0.0%"

    def test_a_missing_value_is_marked_absent(self) -> None:
        assert text.percent(None) == "n/a"

    def test_a_value_above_a_hundred_keeps_its_bound(self) -> None:
        assert text.percent(100.004) == "<100.1%"


class TestShare:
    def test_a_fraction_becomes_a_percentage(self) -> None:
        assert text.share(0.8027) == "80.3%"

    def test_a_whole_share_is_a_hundred(self) -> None:
        assert text.share(1.0) == "100.0%"

    def test_a_share_just_short_keeps_its_bound(self) -> None:
        """9,999 columns of 10,000 is not every column.

        At one decimal place it rounds to 100.0%, so the bound is written
        instead.
        """
        assert text.share(9999 / 10000) == ">99.9%"

    def test_a_missing_share_is_absent(self) -> None:
        assert text.share(None) == "n/a"


class TestScore:
    def test_a_whole_score_has_no_decimal_point(self) -> None:
        """Matrix entries are whole numbers.

        Writing 429 as 429.00 invites a reader to look for precision that is
        not there.
        """
        assert text.score(429.0) == "429"

    def test_a_fractional_score_keeps_one_place(self) -> None:
        assert text.score(3296.5) == "3,296.5"

    def test_large_scores_are_grouped(self) -> None:
        assert text.score(12345.0) == "12,345"

    def test_a_negative_score_is_written(self) -> None:
        assert text.score(-18.0) == "-18"

    def test_a_missing_score_is_absent(self) -> None:
        assert text.score(None) == "n/a"


class TestCounts:
    def test_thousands_are_grouped(self) -> None:
        assert text.thousands(1234567) == "1,234,567"

    def test_one_of_something_is_singular(self) -> None:
        assert text.plural(1, "run") == "1 run"

    def test_several_are_plural(self) -> None:
        assert text.plural(3, "run") == "3 runs"

    def test_none_is_plural(self) -> None:
        assert text.plural(0, "run") == "0 runs"

    def test_a_custom_suffix_is_used(self) -> None:
        assert text.plural(2, "match", "es") == "2 matches"

    def test_a_bounded_count_says_it_is_a_bound(self) -> None:
        assert text.count_with_bound(1000, True) == "at least 1,000"
        assert text.count_with_bound(1000, False) == "1,000"


class TestResidues:
    def test_nucleotide_lengths_are_in_bases(self) -> None:
        assert text.residues(628, DNA) == "628 bp"

    def test_protein_lengths_are_in_residues(self) -> None:
        """A protein length written 'bp' makes a reader doubt the rest."""
        assert text.residues(147, PROTEIN) == "147 aa"


class TestRanges:
    def test_an_interval_is_written_with_a_dash(self) -> None:
        assert text.position_range(1, 147) == "1-147"

    def test_a_large_interval_is_grouped(self) -> None:
        assert text.position_range(1, 4679) == "1-4,679"

    def test_an_empty_interval_is_a_dash(self) -> None:
        """A local alignment of nothing has no interval, not interval zero."""
        assert text.position_range(0, 0) == "-"


class TestTruncate:
    def test_a_short_label_passes_through(self) -> None:
        assert text.truncate("short", 10) == "short"

    def test_a_long_label_is_cut_with_an_ellipsis(self) -> None:
        cut = text.truncate("a" * 20, 10)
        assert len(cut) == 10
        assert cut.endswith("\u2026")


class TestNumber:
    def test_a_number_is_written_to_the_places_asked_for(self) -> None:
        assert text.number(1.23456, 3) == "1.235"

    def test_a_missing_number_is_absent(self) -> None:
        assert text.number(None) == "n/a"
