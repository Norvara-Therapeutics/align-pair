"""The README held to the program, rather than to a reading of it.

The quoted summary blocks are re-run through the command line and compared
byte for byte, the mode table is reconstructed from the code, and every flag
named anywhere in the prose is checked against the parser. A README that has
drifted from the program fails here rather than misleading a reader.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from align_pair.aligner import MODES
from align_pair.cli import build_parser, main

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def blocks() -> list[tuple[str, str]]:
    """Each bash command in the README paired with the output quoted after it."""
    pattern = re.compile(
        r"```bash\n([^`]*?)\n```\n\n```\n([^`]*?)```", re.DOTALL
    )
    return [(command, output) for command, output in pattern.findall(README)]


def options_section() -> str:
    """The whole README except the sections that quote another program's flags.

    'Install' carries pip's own --pre, so a flag named there is not this
    program's. Everything else is checked, which is what catches a flag named
    in passing in prose.
    """
    kept: list[str] = []
    skipping = False
    for line in README.splitlines():
        if line.startswith("## "):
            skipping = line.strip() == "## Install"
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


class TestQuotedOutput:
    def test_there_are_quoted_blocks_to_check(self) -> None:
        assert len(blocks()) >= 2

    @pytest.mark.parametrize("index", range(2))
    def test_a_quoted_block_is_what_the_command_prints(
        self, index: int, capsys
    ) -> None:
        command, expected = blocks()[index]
        arguments = command.split()
        assert arguments[0] == "align-pair"
        resolved = [
            str(ROOT / part) if part.startswith("examples/") else part
            for part in arguments[1:]
        ]
        code = main(resolved)
        printed = capsys.readouterr().out
        assert code == 0
        # The README quotes the relative paths a reader would type.
        assert printed == expected

    def test_the_report_command_runs(self, tmp_path, capsys) -> None:
        command = next(
            text for text, _output in _commands() if "--report" in text
        )
        arguments = [
            str(ROOT / part) if part.startswith("examples/") else part
            for part in command.split()[1:]
        ]
        arguments[arguments.index("--report") + 1] = str(tmp_path / "out.html")
        assert main(arguments) == 0
        capsys.readouterr()
        assert (tmp_path / "out.html").exists()


class TestModeTable:
    def test_every_mode_is_in_the_table(self) -> None:
        table = re.search(r"## The modes\n\n(.*?)\n\n", README, re.DOTALL).group(1)
        named = re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE)
        assert named == list(MODES)

    def test_no_mode_is_described_twice(self) -> None:
        table = re.search(r"## The modes\n\n(.*?)\n\n", README, re.DOTALL).group(1)
        named = re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE)
        assert len(set(named)) == len(named)


class TestFlags:
    def _parser_flags(self) -> set[str]:
        found: set[str] = set()
        parser = build_parser()
        stack = [parser]
        while stack:
            current = stack.pop()
            for action in current._actions:
                found.update(action.option_strings)
                if hasattr(action, "choices") and action.choices:
                    for choice in action.choices.values() if isinstance(
                        action.choices, dict
                    ) else []:
                        stack.append(choice)
        return found

    def test_every_flag_named_in_the_readme_exists(self) -> None:
        named = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", options_section()))
        unknown = named - self._parser_flags()
        assert unknown == set(), f"the README names flags the parser does not: {unknown}"

    def test_the_check_can_fail(self) -> None:
        """A flag audit that cannot fail proves nothing."""
        assert "--nonexistent-flag" not in self._parser_flags()

    def test_every_output_flag_is_documented(self) -> None:
        for flag in ("--output", "--columns", "--alignment", "--report"):
            assert flag in README


class TestClaims:
    def test_the_gap_convention_is_stated_as_the_code_implements_it(self) -> None:
        assert "`open + (L - 1) * extend`" in README

    def test_the_quoted_speed_is_labelled_as_measured(self) -> None:
        assert "Measured on one core" in README

    def test_the_example_accessions_are_the_files_on_disk(self) -> None:
        from align_pair import reader

        quoted = set(re.findall(r"`([A-Z]{2,3}_?\d+\.\d)`", README))
        on_disk = {
            reader.read_all(path)[0].identifier
            for path in sorted((ROOT / "examples").glob("*.fasta"))
        }
        assert on_disk <= quoted

    def test_the_license_is_named(self) -> None:
        assert "PolyForm Noncommercial License 1.0.0" in README

    def test_the_distribution_name_is_the_one_that_installs(self) -> None:
        configured = tomllib.load((ROOT / "pyproject.toml").open("rb"))
        assert configured["project"]["name"] in README


def _commands() -> list[tuple[str, str]]:
    pattern = re.compile(r"```bash\n([^`]*?)\n```", re.DOTALL)
    return [(command, "") for command in pattern.findall(README)]


def test_the_readme_opens_with_the_project_name() -> None:
    assert README.startswith("# align-pair\n")


def test_the_readme_has_not_grown_past_what_it_needs() -> None:
    """A README nobody finishes is a README nobody reads.

    The limit is deliberately generous; its purpose is to make growth visible
    rather than to forbid it.
    """
    assert len(README) < 14_000, f"the README is {len(README)} bytes"


def test_standard_input_is_offered(capsys) -> None:
    """The README says '-' reads standard input, so it has to."""
    assert "'-' for standard input" in README
    text = (ROOT / "examples" / "hbb_human.fasta").read_text()
    import align_pair.reader as reader

    records = reader.read_all(io.StringIO(text))
    assert records[0].identifier == "NP_000509.1"
