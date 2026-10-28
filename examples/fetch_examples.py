"""Re-download the example records from the NCBI.

The files beside this script are real records, downloaded unmodified, so that
the figures in the README and in ``tests/test_reference_examples.py`` are about
real sequence rather than about something invented to make a point. Run this to
replace them:

    python examples/fetch_examples.py

The records are two pairs of homologues and one pair of their coding mRNAs: the
VP1 capsid proteins of adeno-associated virus serotypes 2 and 9, the beta
globins of human and mouse, and the globin mRNAs, which share a coding sequence
and differ in their untranslated regions - which is what makes them worth
aligning globally and locally and comparing the two.
"""

from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

ENTREZ = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: File name to the accession and the database it is in.
RECORDS: dict[str, tuple[str, str]] = {
    "aav2_vp1.fasta": ("YP_680426.1", "protein"),
    "aav9_vp1.fasta": ("AAS99264.1", "protein"),
    "hbb_human.fasta": ("NP_000509.1", "protein"),
    "hbb_mouse.fasta": ("NP_032246.2", "protein"),
    "hbb_human_mrna.fasta": ("NM_000518.5", "nuccore"),
    "hbb_mouse_mrna.fasta": ("NM_008220.5", "nuccore"),
}

#: The NCBI asks for no more than three requests a second without a key.
PAUSE = 0.4


def fetch(accession: str, database: str) -> str:
    """Download one record as FASTA."""
    query = urllib.parse.urlencode(
        {
            "db": database,
            "id": accession,
            "rettype": "fasta",
            "retmode": "text",
            "tool": "align-pair",
        }
    )
    with urllib.request.urlopen(f"{ENTREZ}?{query}", timeout=60) as response:
        return response.read().decode("ascii")


def main() -> int:
    for filename, (accession, database) in RECORDS.items():
        text = fetch(accession, database).rstrip("\n") + "\n"
        if not text.startswith(">"):
            print(f"{accession}: not a FASTA record", file=sys.stderr)
            return 1
        residues = "".join(line.strip() for line in text.splitlines()[1:])
        (HERE / filename).write_text(text, encoding="utf-8")
        print(f"{filename}: {accession}, {len(residues)} residues")
        time.sleep(PAUSE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
