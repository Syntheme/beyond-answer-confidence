"""Build the blog pages from the sources in ``src/``.

Each page is one self-contained HTML file: text, styles, chart data and chart
code are all inlined, so it can be opened or published as it is.

Usage (from the repository root, standard library only)::

    python assets/blog/build.py           # write the pages
    python assets/blog/build.py --check   # fail if a page is out of date

The builder also numbers sections and footnotes per page, writes each page's
table of contents and checks that every footnote, link target and chart slot
resolves.
"""

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
CONTENT = SRC / "content"
CHARTS = SRC / "charts"


@dataclass(frozen=True)
class Page:
    """One output page.

    Attributes:
        filename: Output file name, next to this script.
        title: Contents of ``<title>``.
        description: Contents of the meta description.
        fragments: Files in ``src/content/`` joined, in order, into the body.
    """

    filename: str
    title: str
    description: str
    fragments: tuple[str, ...]


TITLE = "Does a decision model (like Jev) know when it is guessing?"

PAGES = (
    Page(
        "index.html",
        TITLE,
        "We ran about 575,000 queries against Jev, a decision model. Its confidence "
        "worked on familiar tasks, but asking it whether it knows mostly picked up "
        "surface clues.",
        (
            "header-full.html",
            "intro.html",
            "sections-1.html",
            "sections-2.html",
            "conclusion.html",
            "methods.html",
        ),
    ),
    Page(
        "part-1.html",
        f"{TITLE} (Post 1)",
        "We ran about 575,000 queries against Jev, a decision model. "
        "Its confidence worked on familiar tasks and stayed high when it had nothing "
        "to go on.",
        (
            "header-part-1.html",
            "intro.html",
            "sections-1.html",
            "end-part-1.html",
            "methods.html",
        ),
    ),
    Page(
        "part-2.html",
        f"{TITLE} (Post 2)",
        "We asked Jev, a decision model, whether it knows the answer. Its replies "
        "mostly picked up surface clues; questions about the case itself held up.",
        (
            "header-part-2.html",
            "intro-part-2.html",
            "sections-2.html",
            "conclusion.html",
            "methods.html",
        ),
    ),
)

NOTE_REF = re.compile(r'<sup data-note="([a-z0-9-]+)"></sup>')
NOTE_DEF = re.compile(r'<li data-note="([a-z0-9-]+)">\s*(.*?)\s*</li>', re.S)
HEADING = re.compile(r'<h2 id="([^"]+)">(.*?)</h2>', re.S)
EYEBROW = re.compile(r'<span class="eyebrow">(.*?)</span>', re.S)
CHART = re.compile(r'<div class="chart" id="([A-Za-z0-9]+)"></div>')


class BuildError(Exception):
    """A source problem that stops the build."""


def read(path: Path) -> str:
    """Read a UTF-8 text file.

    Args:
        path: File to read.

    Returns:
        The file's text.
    """
    return path.read_text(encoding="utf-8")


def load_notes() -> dict[str, str]:
    """Footnote bodies by key, from ``src/content/notes.html``.

    Returns:
        Mapping from note key to its HTML body.

    Raises:
        BuildError: If a key is defined twice.
    """
    notes: dict[str, str] = {}
    for key, body in NOTE_DEF.findall(read(CONTENT / "notes.html")):
        if key in notes:
            raise BuildError(f"footnote {key!r} is defined twice")
        notes[key] = body
    return notes


def number_sections(body: str) -> tuple[str, str]:
    """Number the eyebrows of the ``<h2>`` headings and build the contents.

    Args:
        body: Page body.

    Returns:
        The body with ``01 · `` style prefixes added, and the contents box.
    """
    entries: list[str] = []
    count = 0

    def heading(m: re.Match[str]) -> str:
        nonlocal count
        hid, inner = m.group(1), m.group(2)
        label = " ".join(re.sub(r"<[^>]+>", "", EYEBROW.sub("", inner)).split())
        entries.append(f'        <li><a href="#{hid}">{label}</a></li>')
        if EYEBROW.search(inner):
            count += 1
            inner = EYEBROW.sub(
                lambda e: f'<span class="eyebrow">{count:02d} · {e.group(1)}</span>',
                inner,
                count=1,
            )
        return f'<h2 id="{hid}">{inner}</h2>'

    body = HEADING.sub(heading, body)
    nav = (
        '    <div class="toc" role="navigation" aria-label="Contents">'
        "<strong>Contents</strong>\n"
        "      <ol>\n" + "\n".join(entries) + "\n      </ol>\n    </div>"
    )
    return body, nav


def number_notes(body: str, notes: dict[str, str]) -> tuple[str, str, set[str]]:
    """Replace footnote markers with numbered links, in reading order.

    A note cited more than once keeps its first number.

    Args:
        body: Page body.
        notes: Footnote bodies by key.

    Returns:
        The body, the ``<li>`` items of the notes list, and the keys used.

    Raises:
        BuildError: If a marker names an undefined note.
    """
    order: list[str] = []

    def ref(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in notes:
            raise BuildError(f"footnote {key!r} is cited but not defined")
        first = key not in order
        if first:
            order.append(key)
        n = order.index(key) + 1
        anchor = f' id="r{n}"' if first else ""
        return f'<sup><a href="#n{n}"{anchor}>{n}</a></sup>'

    body = NOTE_REF.sub(ref, body)
    items = "\n".join(
        f'      <li id="n{n}">{notes[key]} <a href="#r{n}" aria-label="Back">↩</a></li>'
        for n, key in enumerate(order, 1)
    )
    return body, items, set(order)


def blob(path: Path) -> str:
    """Chart data as compact JSON, safe to inline in a ``<script>`` element.

    Args:
        path: JSON file.

    Returns:
        Compact JSON with ``</`` escaped.
    """
    data = json.loads(read(path))
    return json.dumps(data, separators=(",", ":")).replace("</", "<\\/")


def check_page(name: str, page: str, chart_code: str) -> None:
    """Check that ids, link targets and chart slots resolve.

    Args:
        name: Page file name, for messages.
        page: Full page HTML.
        chart_code: The chart scripts, to look chart slots up in.

    Raises:
        BuildError: On a leftover placeholder, a duplicate id, a link to a
            missing id, or a chart slot that no script draws.
    """
    if "{{" in page:
        left = sorted(set(re.findall(r"\{\{\s*[A-Z0-9]+\s*\}\}", page)))
        raise BuildError(f"{name}: unfilled placeholder(s) {left}")
    ids = re.findall(r'\sid="([^"]+)"', page)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise BuildError(f"{name}: duplicate id(s) {dupes}")
    missing = sorted(set(re.findall(r'href="#([^"]+)"', page)) - set(ids))
    if missing:
        raise BuildError(f"{name}: links to missing id(s) {missing}")
    for chart in CHART.findall(page):
        if f'"{chart}"' not in chart_code:
            raise BuildError(f"{name}: no chart script draws {chart!r}")


def assemble(p: Page, notes: dict[str, str]) -> tuple[str, str, set[str]]:
    """Join a page's fragments and number its sections and footnotes.

    Args:
        p: The page.
        notes: Footnote bodies by key.

    Returns:
        The body (with its contents box), the ``<li>`` items of the
        notes list, and the note keys used.

    Raises:
        BuildError: If the body lacks its single ``{{TOC}}`` or cites an
            undefined note.
    """
    body = "\n".join(read(CONTENT / f).rstrip("\n") for f in p.fragments)
    body, nav = number_sections(body)
    if body.count("{{TOC}}") != 1:
        raise BuildError(f"{p.filename}: needs exactly one {{{{TOC}}}}")
    body = body.replace("{{TOC}}", nav)
    return number_notes(body, notes)


def build() -> dict[str, str]:
    """Assemble every page.

    Returns:
        Page HTML by output file name.

    Raises:
        BuildError: If the sources are inconsistent.
    """
    notes = load_notes()
    skeleton = read(SRC / "page.html")
    style = read(SRC / "style.css").rstrip()
    charts = {i: read(CHARTS / f"charts-{i}.js").rstrip() for i in (1, 2, 3)}
    data = {i: blob(CHARTS / f"data-{i}.json") for i in (1, 2, 3)}
    chart_code = "\n".join(charts.values())
    used: set[str] = set()
    out: dict[str, str] = {}
    for p in PAGES:
        body, items, keys = assemble(p, notes)
        used |= keys
        page = skeleton
        for ph, value in (
            ("TITLE", html.escape(p.title)),
            ("DESCRIPTION", html.escape(p.description)),
            ("STYLE", style),
            ("BODY", body),
            ("NOTES", items),
            *((f"DATA{i}", data[i]) for i in (1, 2, 3)),
            *((f"CHARTS{i}", charts[i]) for i in (1, 2, 3)),
        ):
            if page.count(f"{{{{{ph}}}}}") != 1:
                raise BuildError(f"page.html needs exactly one {{{{{ph}}}}}")
            page = page.replace(f"{{{{{ph}}}}}", value)
        check_page(p.filename, page, chart_code)
        out[p.filename] = page
    unused = sorted(set(notes) - used)
    if unused:
        raise BuildError(f"footnote(s) defined but never cited: {unused}")
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if a page is out of date"
    )
    args = parser.parse_args(argv)
    try:
        pages = build()
    except BuildError as err:
        sys.stderr.write(f"blog build: {err}\n")
        return 2
    stale = []
    for name, text in pages.items():
        path = HERE / name
        if args.check:
            if not path.exists() or read(path) != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            rel = path.relative_to(HERE.parent.parent)
            sys.stdout.write(f"wrote {rel} ({len(text) // 1024} KB)\n")
    if stale:
        sys.stderr.write(
            f"blog build: out of date: {', '.join(stale)}; "
            "run python assets/blog/build.py\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
