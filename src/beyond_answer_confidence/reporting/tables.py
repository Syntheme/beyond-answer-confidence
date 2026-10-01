"""Markdown tables drawn from experiment outputs.

Each builder takes the parsed ``summary.json`` (and rows where needed) and
returns Markdown; :data:`TABLES` maps experiment names to functions that
read the outputs and write ``<output_dir>/<name>/tables.md``.
"""

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from beyond_answer_confidence.experiments.base import read_jsonl
from beyond_answer_confidence.settings import Settings


def fmt(value: Any, digits: int = 3) -> str:
    """Format one cell.

    Args:
        value: Number, string, interval (two-element list) or ``None``.
        digits: Decimal places for floats.

    Returns:
        The cell text (``-`` for missing or NaN values).
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.{digits}f}"
    if isinstance(value, list | tuple) and len(value) == 2:
        return f"[{fmt(value[0], digits)}, {fmt(value[1], digits)}]"
    return str(value).replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a Markdown table.

    Args:
        headers: Column headers.
        rows: Cell values per row (formatted with :func:`fmt`).

    Returns:
        The table, ending with a newline.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _section(title: str, table: str) -> str:
    return f"## {title}\n\n{table}\n"


def _sorted_int(d: Mapping[str, Any]) -> list[str]:
    return sorted(d, key=lambda k: int(k))


def synthetic_worlds_tables(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-cell means of the synthetic-worlds rows (main arm).

    Args:
        rows: Synthetic-worlds rows.

    Returns:
        Markdown.
    """
    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("alone"):
            cells[str(r["cell"])].append(r)

    def mean(g: list[Mapping[str, Any]], key: str) -> float | None:
        v = [float(r[key]) for r in g if r.get(key) is not None]
        return sum(v) / len(v) if v else None

    table = markdown_table(
        ["cell", "n", "mean TV from ideal", "mean top probability", "mean P(settled)"],
        [
            [c, len(g), mean(g, "tv"), mean(g, "p_max"), mean(g, "p_settled")]
            for c, g in sorted(cells.items())
        ],
    )
    return _section("Distance from the ideal distribution by cell", table)


def knowledge_boundary_tables(results: Mapping[str, Any]) -> str:
    """PopQA by popularity quintile and fabricated entities.

    Args:
        results: The ``results`` block of the summary.

    Returns:
        Markdown.
    """
    q = results.get("popqa_by_quintile", {})
    out = _section(
        "PopQA by popularity quintile",
        markdown_table(
            [
                "quintile",
                "popularity range",
                "n",
                "accuracy",
                "mean top prob.",
                "mean P(known)",
            ],
            [
                [
                    k,
                    f"{q[k]['s_pop_range'][0]:,}-{q[k]['s_pop_range'][1]:,}",
                    q[k]["n"],
                    q[k]["accuracy"],
                    q[k]["mean_p_max"],
                    q[k]["mean_p_known"],
                ]
                for k in _sorted_int(q)
            ],
        ),
    )
    if fab := results.get("fabricated"):
        out += _section(
            "Fabricated entities (no answer is correct)",
            markdown_table(
                ["n", "mean top prob.", "share top prob. >= 0.5", "mean P(known)"],
                [
                    [
                        fab["n"],
                        fab["mean_p_max"],
                        fab["share_p_max_ge_0.5"],
                        fab["mean_p_known"],
                    ]
                ],
            ),
        )
    return out


def evidence_sufficiency_tables(results: Mapping[str, Any]) -> str:
    """HotpotQA by evidence cell and Quizbowl by clues shown.

    Args:
        results: The ``results`` block of the summary.

    Returns:
        Markdown.
    """
    h = results.get("hotpot_by_cell", {})
    k = results.get("quizbowl_by_k", {})
    cols = ["n", "accuracy", "mean_p_max", "mean_p_enough"]
    heads = ["n", "accuracy", "mean top prob.", "mean P(enough)"]
    return _section(
        "HotpotQA by evidence cell",
        markdown_table(["cell", *heads], [[c, *(h[c][x] for x in cols)] for c in h]),
    ) + _section(
        "Quizbowl by sentences shown",
        markdown_table(
            ["k", *heads], [[x, *(k[x][c] for c in cols)] for x in _sorted_int(k)]
        ),
    )


def _calibration_rows(blocks: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            name,
            b.get("n"),
            b.get("accuracy"),
            b.get("mean_confidence"),
            b.get("smece"),
            b.get("ci"),
            b.get("error_auroc_p_max"),
        ]
        for name, b in blocks.items()
    ]


CALIBRATION_HEADERS = (
    "set",
    "n",
    "accuracy",
    "mean confidence",
    "SmoothECE",
    "95 % interval",
    "error AUROC (top prob.)",
)


def benchmarks_tables(results: Mapping[str, Any]) -> str:
    """Calibration per benchmark.

    Args:
        results: The ``results`` block of the summary.

    Returns:
        Markdown.
    """
    return _section(
        "Calibration of the top-label probability",
        markdown_table(
            CALIBRATION_HEADERS, _calibration_rows(results.get("calibration", {}))
        ),
    )


def option_count_tables(results: Mapping[str, Any]) -> str:
    """Calibration per multiple-choice set and per option count.

    Args:
        results: The ``results`` block of the summary.

    Returns:
        Markdown.
    """
    sets = {
        name.removesuffix("_calibration"): block
        for name, block in results.items()
        if name.endswith("_calibration") and isinstance(block, Mapping)
    }
    counts = results.get("calibration_by_option_count", {}).get("counts", {})
    per_k = {f"{k} options": counts[k] for k in _sorted_int(counts)}
    return _section(
        "Calibration per multiple-choice set",
        markdown_table(CALIBRATION_HEADERS, _calibration_rows(sets)),
    ) + _section(
        "Calibration by number of intent options",
        markdown_table(CALIBRATION_HEADERS, _calibration_rows(per_k)),
    )


def reliability_tables(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Reliability bins per dataset and condition.

    Args:
        results: Blocks keyed by dataset, each with ``reliability_bins``
            (``{condition: bins}`` or ``{condition: {series: bins}}``).

    Returns:
        Markdown.
    """
    out = ""
    for dataset, block in results.items():
        rows: list[list[Any]] = []
        for cond, value in block.get("reliability_bins", {}).items():
            series = value if isinstance(value, Mapping) else {"": value}
            for sname, bins in series.items():
                rows += [
                    [
                        cond,
                        sname or "-",
                        i + 1,
                        b.get("n"),
                        b["confidence"],
                        b["accuracy"],
                    ]
                    for i, b in enumerate(bins)
                ]
        out += _section(
            f"Reliability bins, {dataset}",
            markdown_table(
                ["condition", "series", "bin", "n", "mean confidence", "accuracy"], rows
            ),
        )
    return out


def _summary(settings: Settings, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (settings.out(name) / "summary.json").read_text(encoding="utf-8")
    )
    return data


def _write(settings: Settings, name: str, title: str, body: str) -> Path:
    path = settings.out(name) / "tables.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}", encoding="utf-8")
    return path


def _results_writer(
    name: str, title: str, build: Callable[[Mapping[str, Any]], str]
) -> Callable[[Settings], list[Path]]:
    def write(settings: Settings) -> list[Path]:
        res = _summary(settings, name).get("results")
        if res is None:
            return []
        return [_write(settings, name, title, build(res))]

    return write


def _blocks(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        k: v
        for k, v in summary.items()
        if isinstance(v, Mapping) and "reliability_bins" in v
    }


def _synthetic(settings: Settings) -> list[Path]:
    rows = read_jsonl(settings.out("synthetic_worlds") / "rows.jsonl")
    return [
        _write(
            settings,
            "synthetic_worlds",
            "Synthetic worlds",
            synthetic_worlds_tables(rows),
        )
    ]


def _errors(settings: Settings) -> list[Path]:
    body = reliability_tables(_blocks(_summary(settings, "error_detection")))
    return [_write(settings, "error_detection", "Error detection", body)]


def _comparator(settings: Settings) -> list[Path]:
    # The comparison writes its summary under the classifier's directory.
    body = reliability_tables(_blocks(_summary(settings, "gliner")))
    return [_write(settings, "gliner", "API vs local classifier", body)]


TABLES: dict[str, Callable[[Settings], list[Path]]] = {
    "synthetic_worlds": _synthetic,
    "knowledge_boundary": _results_writer(
        "knowledge_boundary", "Knowledge boundary", knowledge_boundary_tables
    ),
    "evidence_sufficiency": _results_writer(
        "evidence_sufficiency", "Evidence sufficiency", evidence_sufficiency_tables
    ),
    "benchmarks": _results_writer("benchmarks", "Benchmarks", benchmarks_tables),
    "option_count": _results_writer(
        "option_count", "Option count", option_count_tables
    ),
    "error_detection": _errors,
    "gliner_comparison": _comparator,
}
"""Experiment or analysis name -> writes its tables from the outputs."""
