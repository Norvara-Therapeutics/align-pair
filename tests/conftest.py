"""Fixtures shared by the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_pair import matrices  # noqa: E402
from align_pair.aligner import Gaps  # noqa: E402


@pytest.fixture(scope="session")
def examples() -> Path:
    """The directory holding the example records."""
    return ROOT / "examples"


@pytest.fixture(scope="session")
def blosum62():
    """The default protein matrix."""
    return matrices.load("BLOSUM62")


@pytest.fixture(scope="session")
def nuc44():
    """The default nucleic-acid matrix."""
    return matrices.load("NUC.4.4")


@pytest.fixture(scope="session")
def simple_dna():
    """A match/mismatch matrix over the nucleotide codes."""
    return matrices.simple(1.0, -1.0, "nucleotide")


@pytest.fixture(scope="session")
def simple_protein():
    """A match/mismatch matrix over the amino-acid codes."""
    return matrices.simple(1.0, -1.0, "protein")


@pytest.fixture(scope="session")
def gaps() -> Gaps:
    """A gap penalty whose arithmetic is easy to check by hand."""
    return Gaps(2.0, 1.0)


def write(path: Path, *records: tuple[str, str]) -> Path:
    """Write records as FASTA and return the path."""
    text = "".join(f">{name}\n{sequence}\n" for name, sequence in records)
    path.write_text(text, encoding="utf-8")
    return path
