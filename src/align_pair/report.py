"""The self-contained HTML report.

The document embeds its own style rules and its own charts, loads nothing from
the network and runs no script, so it opens from a local disk, an email
attachment or an archive years from now and looks the same. Output is
byte-deterministic for a given input and set of parameters: nothing in it
depends on the clock, the filesystem or the order of a set.

The alignment itself is shown as the command line shows it, in blocks with
their coordinates, with every column coloured by what it is - identical,
similar, differing, or a gap - and with the marker line kept, so the document
is readable in black and white and by a reader who cannot distinguish the
colours. Each table can be downloaded as the tab-separated file the command
line would have written, carried as a data URI that needs no server.
"""

from __future__ import annotations

import base64
from html import escape
from typing import Sequence

from align_pair.aligner import MODE_SUMMARY, Gaps
from align_pair.alphabet import GAP
from align_pair.analysis import Analysis, Pair
from align_pair.matrices import Matrix, describe, grid
from align_pair.pairview import LEGEND, blocks
from align_pair.rows import (
    COLUMN_COLUMNS,
    PAIR_COLUMNS,
    column_rows,
    pair_rows,
    render,
)
from align_pair.text import (
    plural,
    position_range,
    residues,
    score as score_text,
    share,
    thousands,
    truncate,
)

#: A table larger than this is not embedded for download. Three megabytes of
#: base64 is already a document no mail server will carry, and the file it
#: would hold is one ``--output`` writes in a second.
MAX_EMBEDDED: int = 3_000_000

#: How wide a block of the alignment is in the report.
BLOCK_WIDTH: int = 60

#: The CSS class for each kind of column.
_CLASSES: dict[str, str] = {
    "|": "same",
    ":": "like",
    ".": "diff",
    " ": "gap",
}

_STYLE = """
html { -webkit-text-size-adjust: 100%; }
body { font: 16px/1.55 Georgia, 'Times New Roman', serif; color: #1a1a1a;
       background: #fff; margin: 0 auto; padding: 3rem 1.25rem 4rem;
       max-width: 54rem; }
h1 { font-size: 1.5rem; font-weight: normal; margin: 0 0 .25rem; }
h2, h3 { font-size: .8rem; font-variant: small-caps; letter-spacing: .08em;
     font-weight: normal; margin: 2.5rem 0 .6rem; border-bottom: 1px solid #1a1a1a;
     padding-bottom: .2rem; }
h3 { margin-top: 1.8rem; border-bottom: none; color: #555; }
p { margin: .5rem 0; }
.lede { color: #555; font-size: .9rem; margin-bottom: .5rem; }
.note { color: #555; font-size: .85rem; }
code, .mono { font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
table { border-collapse: collapse; width: 100%; margin: .6rem 0 .2rem;
        font-size: .9rem; }
th { text-align: left; font-weight: normal; font-variant: small-caps;
     letter-spacing: .04em; border-bottom: 1px solid #1a1a1a;
     padding: .25rem .5rem .25rem 0; }
td { padding: .22rem .5rem .22rem 0; border-bottom: 1px solid #e6e6e6;
     vertical-align: baseline; }
td.num, th.num { text-align: right; padding-right: 1rem; font-size: .85rem;
     font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
dl.metrics { display: grid; grid-template-columns: max-content 1fr;
             gap: .15rem 1.2rem; margin: .6rem 0; font-size: .9rem; }
dl.metrics dt { color: #555; }
dl.metrics dd { margin: 0;
     font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
.caveat { border-left: 2px solid #8a5a00; padding: .1rem 0 .1rem .7rem;
          margin: .8rem 0; color: #6b4700; font-size: .85rem; }
.caveat p { margin: .25rem 0; }
details { margin: .4rem 0; }
summary { cursor: pointer; font-size: .85rem; color: #555; }
summary:hover { color: #1a1a1a; }
.download { font-size: .8rem; }
pre.view { font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
     font-size: 12.5px; line-height: 1.45; overflow-x: auto; margin: .6rem 0;
     padding: .6rem .7rem; background: #fafafa; border: 1px solid #e6e6e6; }
pre.view .same { color: #1a1a1a; background: #dcebe6; }
pre.view .like { color: #0f6f6a; background: #eef5f3; }
pre.view .diff { color: #8a5a00; background: #f7efe2; }
pre.view .gap  { color: #8a8a8a; background: #f0f0f0; }
pre.view .label { color: #555; }
pre.view .marks { color: #777; }
.key { font-size: .82rem; color: #555; margin: .3rem 0 0; }
.key span { padding: 0 .25rem; margin-right: .15rem;
     font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
.key .same { background: #dcebe6; }
.key .like { background: #eef5f3; color: #0f6f6a; }
.key .diff { background: #f7efe2; color: #8a5a00; }
.key .gap { background: #f0f0f0; color: #8a8a8a; }
svg.chart { display: block; max-width: 100%; height: auto; margin: .6rem 0 .2rem; }
svg.chart text { font: 11px ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
     fill: #555; }
svg.chart .line { fill: none; stroke: #1a1a1a; stroke-width: 1.2; }
svg.chart .axis { stroke: #bbb; stroke-width: 1; }
svg.chart .peak { stroke: #0f6f6a; stroke-width: 1; stroke-dasharray: 4 3; }
footer { margin-top: 3rem; padding-top: .5rem; border-top: 1px solid #e6e6e6;
         color: #555; font-size: .8rem; }
footer p { margin: .35rem 0; }
@media print { body { padding: 0; max-width: none; }
  h2, h3 { page-break-after: avoid; } details { display: block; }
  summary { display: none; } }
"""


def _definitions(rows: Sequence[tuple[str, str]]) -> list[str]:
    parts = ['<dl class="metrics">']
    for name, value in rows:
        parts.append(f"<dt>{escape(name)}</dt><dd>{escape(value)}</dd>")
    parts.append("</dl>")
    return parts


def _download(
    columns: Sequence[str], rows: Sequence[Sequence[object]], filename: str
) -> str:
    """A link that downloads a table, with the table inside the link.

    The file is carried as a base64 data URI, so the link works from a document
    that has been copied, mailed or archived, with nothing to fetch.
    """
    text = render(columns, rows)
    payload = text.encode("utf-8")
    if len(payload) > MAX_EMBEDDED:
        return (
            f'<p class="note download">{escape(filename)} is '
            f"{len(payload) // 1_000_000} MB, too large to embed; write it with "
            "--output</p>"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        f'<p class="note download"><a download="{escape(filename)}" '
        f'href="data:text/tab-separated-values;base64,{encoded}">'
        f"{escape(filename)}</a> &middot; {plural(len(rows), 'row')}</p>"
    )


def _table(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    numeric: Sequence[int] = (),
) -> list[str]:
    parts = ["<table>", "<thead><tr>"]
    for index, column in enumerate(columns):
        kind = ' class="num"' if index in numeric else ""
        parts.append(f"<th{kind}>{escape(str(column))}</th>")
    parts.extend(["</tr></thead>", "<tbody>"])
    for row in rows:
        parts.append("<tr>")
        for index, value in enumerate(row):
            kind = ' class="num"' if index in numeric else ""
            parts.append(f"<td{kind}>{escape(str(value))}</td>")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return parts


def _caveats(notes: Sequence[str]) -> list[str]:
    if not notes:
        return []
    parts = ['<div class="caveat">']
    for note in notes:
        parts.append(f"<p>{escape(note)}</p>")
    parts.append("</div>")
    return parts


def _marker_classes(marks: str) -> list[str]:
    """The CSS class of each column, by what the column is."""
    return [_CLASSES[mark] for mark in marks]


def _coloured(residues_text: str, classes: Sequence[str]) -> str:
    """Residues wrapped in spans, runs of one class to a span.

    One span per residue would quadruple the size of the document for a long
    alignment; runs of the same kind are common, so they are grouped.
    """
    parts: list[str] = []
    run: list[str] = []
    current = ""
    for residue, kind in zip(residues_text, classes):
        if kind != current and run:
            parts.append(f'<span class="{current}">{escape("".join(run))}</span>')
            run = []
        current = kind
        run.append(residue)
    if run:
        parts.append(f'<span class="{current}">{escape("".join(run))}</span>')
    return "".join(parts)


def _view(pair: Pair, matrix: Matrix) -> list[str]:
    """The side-by-side alignment, coloured, as a preformatted block."""
    alignment = pair.alignment
    if alignment.empty:
        return ['<p class="note">No residues are aligned.</p>']
    first_label = truncate(pair.first.identifier, 14)
    second_label = truncate(pair.second.identifier, 14)
    name_width = max(len(first_label), len(second_label))
    reversed_second = (
        pair.metrics.length_second if pair.strand == "minus" else None
    )
    digits = max(
        len(str(alignment.end_first)),
        len(str(reversed_second or alignment.end_second)),
        1,
    )
    lines: list[str] = []
    for index, block in enumerate(
        blocks(alignment, matrix, BLOCK_WIDTH, reversed_second)
    ):
        classes = _marker_classes(block.marks)
        if index:
            lines.append("")
        head = (
            f'<span class="label">{escape(f"{first_label:<{name_width}}")}</span> '
            f'<span class="label">{block.start_first:>{digits}}</span> '
        )
        lines.append(
            head
            + _coloured(block.first, classes)
            + f' <span class="label">{block.end_first:>{digits}}</span>'
        )
        lines.append(
            f'<span class="label">{escape(" " * name_width)}</span> '
            f'<span class="label">{escape(" " * digits)}</span> '
            f'<span class="marks">{escape(block.marks)}</span>'
        )
        tail = (
            f'<span class="label">{escape(f"{second_label:<{name_width}}")}</span> '
            f'<span class="label">{block.start_second:>{digits}}</span> '
        )
        lines.append(
            tail
            + _coloured(block.second, classes)
            + f' <span class="label">{block.end_second:>{digits}}</span>'
        )
    key = " ".join(
        f'<span class="{_CLASSES[mark]}">{escape(mark if mark != " " else "-")}</span>'
        f"{escape(meaning)}"
        for mark, meaning in LEGEND
    )
    return [
        '<pre class="view">' + "\n".join(lines) + "</pre>",
        f'<p class="key">{key}</p>',
    ]


def cumulative(pair: Pair, matrix: Matrix, gaps: Gaps) -> list[tuple[int, float]]:
    """The running score along the alignment, column by column.

    Where the score is earned and where it is spent is the one thing a single
    total cannot show: an alignment that climbs steadily and one that earns
    everything in a short block and then holds on report the same number.

    The last value is the alignment's score, which is what makes this a
    decomposition of the score rather than a picture beside it: each gap run is
    charged where it falls, the opening penalty on its first column and the
    extension penalty on each column after.
    """
    total = 0.0
    track: list[tuple[int, float]] = [(0, 0.0)]
    # Which row the previous gap sat in. A gap in one sequence followed by a
    # gap in the other is two runs and pays to open twice, so a flag that only
    # remembered 'the last column was a gap' would undercharge the second and
    # end the track somewhere other than the score.
    continuing = ""
    for index, (upper, lower) in enumerate(
        zip(pair.alignment.aligned_first, pair.alignment.aligned_second), start=1
    ):
        if upper == GAP or lower == GAP:
            side = "first" if upper == GAP else "second"
            total -= gaps.extend if continuing == side else gaps.open
            continuing = side
        else:
            total += matrix.score(upper, lower)
            continuing = ""
        track.append((index, total))
    return track


def _chart(track: Sequence[tuple[int, float]], width: int = 640, height: int = 120) -> str:
    """The running score as an SVG line, with its peak marked."""
    if len(track) < 2:
        return ""
    columns = track[-1][0]
    values = [value for _index, value in track]
    lowest = min(values)
    highest = max(values)
    if highest == lowest:
        highest = lowest + 1
    left, right, top, bottom = 46, 8, 10, 24
    plot_width = width - left - right
    plot_height = height - top - bottom

    def place(index: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * index / columns if columns else 0)
        y = top + plot_height * (highest - value) / (highest - lowest)
        return x, y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (place(i, v) for i, v in track))
    zero_y = place(0, 0.0)[1] if lowest <= 0 <= highest else None
    peak_index, peak_value = max(track, key=lambda item: item[1])
    peak_x, peak_y = place(peak_index, peak_value)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="running score along the alignment">',
        f"<title>running score along the alignment, ending at "
        f"{values[-1]:g}</title>",
    ]
    if zero_y is not None:
        parts.append(
            f'<line class="axis" x1="{left}" y1="{zero_y:.1f}" '
            f'x2="{width - right}" y2="{zero_y:.1f}"><title>score zero</title></line>'
        )
    parts.append(
        f'<line class="peak" x1="{peak_x:.1f}" y1="{top}" x2="{peak_x:.1f}" '
        f'y2="{top + plot_height}"><title>highest running score {peak_value:g} '
        f"at column {peak_index}</title></line>"
    )
    parts.append(f'<polyline class="line" points="{points}"/>')
    parts.append(f'<text x="4" y="{top + 4:.0f}">{highest:g}</text>')
    parts.append(f'<text x="4" y="{top + plot_height:.0f}">{lowest:g}</text>')
    parts.append(f'<text x="{left}" y="{height - 6}">column 1</text>')
    parts.append(
        f'<text x="{width - right}" y="{height - 6}" text-anchor="end">'
        f"{columns}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _pair_metrics(pair: Pair, analysis: Analysis) -> list[tuple[str, str]]:
    metrics = pair.metrics
    alignment = pair.alignment
    alphabet = analysis.detection.alphabet
    start_second, end_second = pair.span_second
    rows = [
        ("score", score_text(alignment.score)),
        ("length", f"{thousands(metrics.length)} columns"),
        (
            "identity",
            f"{thousands(metrics.identities)} of {thousands(metrics.length)} "
            f"columns, {share(metrics.identity)}",
        ),
        (
            "identity of shorter",
            f"{thousands(metrics.identities)} of {thousands(metrics.shorter)} "
            f"{alphabet.unit}, {share(metrics.identity_of_shorter)}",
        ),
        (
            "similarity",
            f"{thousands(metrics.similarities)} of {thousands(metrics.length)} "
            f"columns, {share(metrics.similarity)}",
        ),
        ("mismatches", thousands(metrics.mismatches)),
        (
            "gaps",
            f"{thousands(metrics.gaps)} columns in "
            f"{plural(metrics.gap_runs, 'run')}, longest "
            f"{thousands(metrics.longest_gap)}, {share(metrics.gap_share)}",
        ),
        (
            pair.first.identifier,
            f"{residues(metrics.length_first, alphabet)}, aligned "
            f"{position_range(alignment.start_first, alignment.end_first)}",
        ),
        (
            pair.second.identifier,
            f"{residues(metrics.length_second, alphabet)}, aligned "
            f"{position_range(start_second, end_second)}"
            + (f" on the {pair.strand} strand" if pair.strand != "plus" else ""),
        ),
        (
            "co-optimal",
            f"{'at least ' if alignment.co_optimal_capped else ''}"
            f"{thousands(alignment.co_optimal)} alignment"
            f"{'' if alignment.co_optimal == 1 else 's'} share this score",
        ),
    ]
    return rows


def _parameters(analysis: Analysis) -> list[tuple[str, str]]:
    settings = analysis.settings
    gaps = settings.gaps
    rows = [
        ("mode", f"{settings.mode} - {MODE_SUMMARY[settings.mode]}"),
        ("matrix", f"{analysis.matrix.name} ({analysis.matrix.origin})"),
        (
            "gap penalty",
            f"{gaps.open:g} to open, {gaps.extend:g} to extend; a gap of L "
            f"residues costs {gaps.open:g} + (L - 1) x {gaps.extend:g}",
        ),
        (
            "alphabet",
            analysis.alphabet_label
            + (" (named)" if analysis.detection.requested else " (detected)"),
        ),
    ]
    if settings.band is not None:
        rows.append(("band", f"{settings.band} diagonals either side"))
    if settings.strand != "plus":
        rows.append(("strand", settings.strand))
    return rows


def _title(analysis: Analysis) -> str:
    if len(analysis.pairs) == 1:
        pair = analysis.pairs[0]
        return f"{pair.first.identifier} aligned with {pair.second.identifier}"
    return f"{plural(len(analysis.pairs), 'alignment')}"


def render_html(analysis: Analysis, version: str) -> str:
    """The whole report as one HTML document."""
    title = _title(analysis)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="lede">{escape(_lede(analysis))}</p>',
        "<h2>Parameters</h2>",
    ]
    parts.extend(_definitions(_parameters(analysis)))
    parts.extend(_caveats(analysis.qualifications))

    for pair in analysis.pairs:
        heading = f"{pair.first.identifier} vs {pair.second.identifier}"
        parts.append(f"<h2>{escape(heading)}</h2>")
        parts.extend(_definitions(_pair_metrics(pair, analysis)))
        parts.extend(_caveats(pair.qualifications))
        track = cumulative(pair, analysis.matrix, analysis.settings.gaps)
        chart = _chart(track)
        if chart:
            parts.append(chart)
        parts.extend(_view(pair, analysis.matrix))
        columns = column_rows(pair, analysis.matrix)
        parts.append("<details><summary>the alignment column by column</summary>")
        parts.append(
            _download(
                COLUMN_COLUMNS,
                columns,
                f"{_slug(pair.first.identifier)}-{_slug(pair.second.identifier)}"
                "-columns.tsv",
            )
        )
        parts.append("</details>")

    parts.append("<h2>Summary table</h2>")
    parts.extend(
        _table(
            ("first", "second", "score", "length", "identity", "similarity", "gaps"),
            [
                [
                    pair.first.identifier,
                    pair.second.identifier,
                    score_text(pair.alignment.score),
                    thousands(pair.metrics.length),
                    share(pair.metrics.identity),
                    share(pair.metrics.similarity),
                    thousands(pair.metrics.gaps),
                ]
                for pair in analysis.pairs
            ],
            numeric=(2, 3, 4, 5, 6),
        )
    )
    parts.append(_download(PAIR_COLUMNS, pair_rows(analysis), "alignments.tsv"))

    parts.append("<h2>The matrix</h2>")
    parts.extend(_definitions(describe(analysis.matrix)))
    if analysis.matrix.comments:
        parts.append(
            '<p class="note">'
            + escape(" ".join(analysis.matrix.comments))
            + "</p>"
        )
    parts.append("<details><summary>the whole table</summary>")
    matrix_columns, matrix_body = grid(analysis.matrix)
    parts.extend(
        _table(
            matrix_columns,
            matrix_body,
            numeric=tuple(range(1, len(matrix_columns))),
        )
    )
    parts.append("</details>")

    parts.extend(
        [
            "<footer>",
            "<p>Positions are 1-based and inclusive. A gap of L residues costs "
            "the opening penalty once and the extension penalty L - 1 times, "
            "which is the convention of EMBOSS needle and water; the BLAST "
            "family charges one extension more.</p>",
            "<p>The alignment follows Needleman and Wunsch (1970) or Smith and "
            "Waterman (1981) by mode, with affine gaps after Gotoh (1982). The "
            "BLOSUM matrices are those of Henikoff and Henikoff (1992) and the "
            "PAM matrices those of Dayhoff, Schwartz and Orcutt (1978), as "
            "published by the NCBI.</p>",
            f"<p>align-pair {escape(version)}</p>",
            "</footer>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return "\n".join(parts)


def _lede(analysis: Analysis) -> str:
    counts = len(analysis.pairs)
    sources = ", ".join(analysis.sources) if analysis.sources else "the input"
    return (
        f"{plural(counts, 'pair')} from {sources}, aligned in "
        f"{analysis.settings.mode} mode with {analysis.matrix.name}."
    )


def _slug(text: str) -> str:
    kept = [character if character.isalnum() else "-" for character in text.lower()]
    slug = "".join(kept).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "record"
