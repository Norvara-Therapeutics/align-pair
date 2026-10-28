"""Number formatting shared by the terminal summary and the HTML report.

Only human-facing output is formatted here. The TSV and JSON outputs carry
unrounded values, so a share that is not quite 100% is never written as 100%
to a file another program will read.
"""

from __future__ import annotations

from align_pair.alphabet import Alphabet


def thousands(value: int) -> str:
    """Group an integer in thousands."""
    return f"{value:,}"


def plural(count: int, singular: str, suffix: str = "s") -> str:
    """Count something, agreeing the noun with the number."""
    return f"{thousands(count)} {singular}{'' if count == 1 else suffix}"


def number(value: float | None, places: int = 2, missing: str = "n/a") -> str:
    """Format an optional number to a fixed number of decimal places."""
    if value is None:
        return missing
    return f"{value:,.{places}f}"


def score(value: float | None, missing: str = "n/a") -> str:
    """Format an alignment score, as an integer when it is one.

    Matrix entries are whole numbers, so a score is fractional only when a gap
    penalty is, and writing 429 as 429.00 invites the reader to look for a
    precision that is not there.
    """
    if value is None:
        return missing
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def percent(value: float | None, places: int = 1, missing: str = "n/a") -> str:
    """Format a percentage, never rounding a near-miss to 0% or 100%.

    A share of 99.98% is written '>99.9%' rather than '100.0%', because an
    alignment reported as 100% identical when one column differs is the kind
    of claim a reader will repeat.
    """
    if value is None:
        return missing
    rounded = round(value, places)
    smallest = 10.0**-places
    if value > 0 and rounded == 0:
        return f"<{smallest:.{places}f}%"
    if value < 100 and rounded == 100:
        return f">{100 - smallest:.{places}f}%"
    if value > 100 and rounded == 100:
        return f"<{100 + smallest:.{places}f}%"
    return f"{value:.{places}f}%"


def share(value: float | None, places: int = 1, missing: str = "n/a") -> str:
    """Format a share held as a fraction as a percentage."""
    if value is None:
        return missing
    return percent(100 * value, places, missing)


def residues(count: int, alphabet: Alphabet) -> str:
    """Write a length in the alphabet's own unit.

    Nucleotide lengths are in bases and protein lengths in residues, because a
    protein length written 'bp' is the kind of slip that makes a reader doubt
    everything else in the output.
    """
    return f"{thousands(count)} {alphabet.unit}"


def position_range(start: int, end: int) -> str:
    """Write an inclusive 1-based interval, or a dash for an empty one."""
    if not start or not end:
        return "-"
    return f"{thousands(start)}-{thousands(end)}"


def count_with_bound(value: int, capped: bool) -> str:
    """Write a count that may have stopped at a cap."""
    return f"at least {thousands(value)}" if capped else thousands(value)


def truncate(text: str, width: int = 40) -> str:
    """Shorten a long identifier for a fixed-width column."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"
