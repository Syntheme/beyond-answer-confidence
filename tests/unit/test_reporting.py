import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest

from beyond_answer_confidence.experiments.base import write_json, write_jsonl
from beyond_answer_confidence.reporting import figures, report, reportable, tables
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks.synthetic import make_scenario

matplotlib.use("Agg")

RNG_SEED = 7


def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    r = rng()
    for i in range(16):
        sc = make_scenario(i, 0)
        for cell in ("SFp", "D0", "D1", "D2", "D3", "SP", "US"):
            p = r.dirichlet(np.ones(len(sc.options)))
            dist = {o: float(x) for o, x in zip(sc.options, p, strict=True)}
            rows.append(
                {
                    "scenario": i,
                    "cell": cell,
                    "alone": False,
                    "profile": sc.profile,
                    "dist": dist,
                    "p_max": max(dist.values()),
                    "tv": float(r.uniform()),
                    "p_settled": float(r.uniform()),
                }
            )
    rows.append({**rows[0], "alone": True})
    return rows


def boundary_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = rng()
    rows: list[dict[str, Any]] = [
        {
            "set": "popqa",
            "s_pop": int(10 * (i + 1)),
            "correct": bool(r.uniform() < 0.5),
            "p_max": float(r.uniform(0.25, 1)),
            "p_known": float(r.uniform()),
        }
        for i in range(50)
    ]
    rows += [
        {"set": "fabricated", "p_max": float(x), "p_known": 0.2, "correct": False}
        for x in r.uniform(0.25, 1, 10)
    ]
    rows += [
        {
            "set": "oracle_tf",
            "month": f"2024-{m:02d}",
            "correct": bool(r.uniform() < 0.7),
            "p_max": float(r.uniform(0.5, 1)),
            "p_known": float(r.uniform()),
        }
        for m in range(1, 13)
        for _ in range(4)
    ]
    quint = {
        str(q): {
            "s_pop_range": [10 * (10 * q + 1), 10 * (10 * q + 10)],
            "n": 10,
            "accuracy": 0.5,
            "mean_p_max": 0.6,
            "mean_p_known": 0.4,
        }
        for q in range(5)
    }
    results = {
        "popqa_by_quintile": quint,
        "fabricated": {
            "n": 10,
            "mean_p_max": 0.6,
            "share_p_max_ge_0.5": 0.7,
            "mean_p_known": 0.2,
        },
        "oracle_change_point": {"first_post_month": "2024-08", "drop": 0.3},
        "calibration_after_cutoff": {"mean": 0.1},
    }
    return rows, results


def evidence_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r = rng()
    rows: list[dict[str, Any]] = [
        {
            "set": "hotpot",
            "cell": c,
            "qid": f"h{q}",
            "correct": bool(r.uniform() < 0.6),
            "p_max": float(r.uniform(0.5, 1)),
            "p_enough": float(r.uniform()),
        }
        for c in figures.HOTPOT_CELLS
        for q in range(6)
    ]
    rows += [
        {
            "set": "quizbowl",
            "k": k,
            "qid": f"q{q}",
            "correct": bool(r.uniform() < 0.2 * k),
            "p_max": float(r.uniform(0.25, 1)),
            "p_enough": float(r.uniform()),
        }
        for k in range(1, 5)
        for q in range(6)
    ]
    block = {"n": 6, "accuracy": 0.5, "mean_p_max": 0.7, "mean_p_enough": 0.4}
    results = {
        "hotpot_by_cell": {c: dict(block) for c in figures.HOTPOT_CELLS},
        "quizbowl_by_k": {str(k): dict(block) for k in range(1, 5)},
    }
    return rows, results


def benchmark_rows() -> list[dict[str, Any]]:
    r = rng()
    rows: list[dict[str, Any]] = [
        {"set": name, "p_max": float(p), "correct": bool(r.uniform() < p)}
        for name in ("triviaqa", "simpleqa", "truthfulqa_binary")
        for p in r.uniform(0.25, 1, 200)
    ]
    rows += [
        {
            "set": "chaos_nli",
            "human_entropy": float(r.uniform(0, np.log2(3))),
            "norm_entropy": float(r.uniform()),
        }
        for _ in range(80)
    ]
    return rows


def option_rows() -> list[dict[str, Any]]:
    """Rows in the option-count format: set, p_max, correct, error_type, K."""
    r = rng()
    rows: list[dict[str, Any]] = []
    for name in ("mmlu_redux", "mmlu_cf", "anli"):
        for p in r.uniform(0.3, 1, 300):
            row = {"set": name, "p_max": float(p), "correct": bool(r.uniform() < p)}
            if name == "mmlu_redux":
                row["error_type"] = "ok" if r.uniform() < 0.9 else "bad_question"
            rows.append(row)
    rows += [
        {"set": "clinc_k", "K": k, "p_max": float(p), "correct": True}
        for k in (5, 10, 50, 150)
        for p in r.uniform(0.8, 1, 20)
    ]
    return rows


def bins(n: int = 5) -> list[dict[str, float]]:
    return [
        {"confidence": (i + 0.5) / n, "accuracy": (i + 0.3) / n, "n": 20}
        for i in range(n)
    ]


def test_intervals() -> None:
    lo, hi = figures.ci_mean([0.2, 0.4, 0.6])
    assert lo < 0.4 < hi
    assert figures.ci_mean([0.5]) == (0.5, 0.5)
    assert all(np.isnan(figures.ci_mean([])))
    lo, hi = figures.ci_prop([True, False, True, True])
    assert 0 <= lo < 0.75 < hi <= 1
    assert all(np.isnan(figures.ci_prop([])))


def test_label_anchor_prefers_clear_points() -> None:
    pts = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.2)]
    other = [[(0.1, 0.12), (0.5, 0.52)]]
    assert figures.label_anchor(pts, other) == (0.9, 0.2)
    assert figures.label_anchor(pts, []) in pts


def test_figures_draw(tmp_path: Path) -> None:
    out = [
        figures.synthetic_worlds_figure(synthetic_rows(), tmp_path / "s.png"),
        figures.knowledge_boundary_figure(*boundary_data(), tmp_path / "k.png"),
        figures.evidence_sufficiency_figure(
            *evidence_data(), tmp_path / "e.png", min_n=1
        ),
        figures.benchmarks_figure(benchmark_rows(), tmp_path / "b.png"),
        figures.option_count_figure(option_rows(), tmp_path / "o.png"),
        figures.intent_reliability_figure(
            {
                "banking77": {"reliability_bins": {"L1": bins(), "Lname": bins()}},
                "other": {"reliability_bins": {"L1": bins(3)}},
            },
            tmp_path / "r.png",
        ),
        figures.comparator_reliability_figure(
            {
                "L0": {"jev": bins(), "gliner": bins()},
                "names": {"jev": [], "gliner": bins()},
                "L1": {"jev": bins(), "gliner": []},
            },
            tmp_path / "c.png",
            "banking77",
        ),
    ]
    for p in out:
        assert p.exists()
        assert p.stat().st_size > 1000


def test_reliability_panels_without_counts(tmp_path: Path) -> None:
    pts = [{"confidence": 0.5, "accuracy": 0.4}, {"confidence": 0.9, "accuracy": 0.8}]
    path = figures.plot_reliability_panels(
        [("a", {"x": pts}), ("b", {"x": pts}), ("c", {"x": pts}), ("d", {})],
        {"x": ("X", figures.S1, "o"), "y": ("Y", figures.S2, "s")},
        "title",
        tmp_path / "p.png",
        ncols=2,
    )
    assert path.exists()


def test_figures_tolerate_missing_sets(tmp_path: Path) -> None:
    rows = [r for r in benchmark_rows() if r["set"] != "chaos_nli"]
    assert figures.benchmarks_figure(rows, tmp_path / "b.png").exists()
    rows = [r for r in option_rows() if r["set"] != "clinc_k"]
    assert figures.option_count_figure(rows, tmp_path / "o.png").exists()


def test_fmt_and_markdown_table() -> None:
    assert tables.fmt(None) == "-"
    assert tables.fmt(True) == "yes"
    assert tables.fmt(False) == "no"
    assert tables.fmt(12345) == "12,345"
    assert tables.fmt(float("nan")) == "-"
    assert tables.fmt(0.12345) == "0.123"
    assert tables.fmt([0.1, 0.2]) == "[0.100, 0.200]"
    assert tables.fmt("a|b") == "a\\|b"
    md = tables.markdown_table(["a", "b"], [[1, 0.5]])
    assert md.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.500 |"]


def test_table_builders() -> None:
    assert "| SFp | 16 |" in tables.synthetic_worlds_tables(synthetic_rows())
    _, kb = boundary_data()
    text = tables.knowledge_boundary_tables(kb)
    assert "Fabricated" in text
    assert "10-100" in text
    _, ev = evidence_data()
    assert "Quizbowl" in tables.evidence_sufficiency_tables(ev)
    cal = {"n": 10, "accuracy": 0.5, "mean_confidence": 0.6, "smece": 0.1,
           "ci": [0.05, 0.15], "error_auroc_p_max": 0.7}  # fmt: skip
    text = tables.benchmarks_tables({"calibration": {"simpleqa": cal}})
    assert "| simpleqa | 10 |" in text
    text = tables.option_count_tables(
        {
            "anli_calibration": cal,
            "calibration_by_option_count": {"counts": {"5": cal, "150": cal}},
        }
    )
    assert "| anli |" in text
    assert "| 5 options |" in text
    assert text.index("5 options") < text.index("150 options")
    text = tables.reliability_tables(
        {
            "banking77": {"reliability_bins": {"L1": bins(2)}},
            "clinc150": {"reliability_bins": {"L0": {"jev": bins(2)}}},
        }
    )
    assert "| L1 | - | 1 |" in text
    assert "| L0 | jev | 2 |" in text


def settings_for(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "out")


def write_outputs(s: Settings) -> None:
    write_jsonl(s.out("synthetic_worlds") / "rows.jsonl", synthetic_rows())
    write_json(s.out("synthetic_worlds") / "summary.json", {"config": {"seed": 0}})
    rows, res = boundary_data()
    write_jsonl(s.out("knowledge_boundary") / "rows.jsonl", rows)
    write_json(s.out("knowledge_boundary") / "summary.json", {"results": res})
    rows, res = evidence_data()
    for block in res["quizbowl_by_k"].values():
        block["n"] = 100
    write_jsonl(s.out("evidence_sufficiency") / "rows.jsonl", rows)
    write_json(s.out("evidence_sufficiency") / "summary.json", {"results": res})
    write_jsonl(s.out("benchmarks") / "rows.jsonl", benchmark_rows())
    write_json(s.out("benchmarks") / "summary.json", {"results": {"calibration": {}}})
    write_jsonl(s.out("option_count") / "rows.jsonl", option_rows())
    write_json(s.out("option_count") / "summary.json", {"results": {}})
    write_json(
        s.out("error_detection") / "summary.json",
        {"config": {}, "banking77": {"reliability_bins": {"L1": bins()}}},
    )
    write_json(
        s.out("gliner") / "summary.json",
        {
            "config": {},
            "banking77": {
                "reliability_bins": {"L0": {"jev": bins(), "gliner": bins()}}
            },
        },
    )


def test_report_every_reporter(tmp_path: Path) -> None:
    s = settings_for(tmp_path)
    write_outputs(s)
    assert set(reportable()) == set(figures.FIGURES) | set(tables.TABLES)
    for name in reportable():
        written = report(name, s)
        assert written, name
        assert all(p.exists() for p in written)
        assert any(p.suffix == ".md" for p in written)
        assert any(p.suffix == ".png" for p in written)
    only_tables = report("benchmarks", s, figures=False)
    assert [p.suffix for p in only_tables] == [".md"]


def test_report_without_results(tmp_path: Path) -> None:
    s = settings_for(tmp_path)
    for name in ("knowledge_boundary", "evidence_sufficiency"):
        write_json(s.out(name) / "summary.json", {"units_complete": 0})
        assert report(name, s) == []
    write_json(s.out("error_detection") / "summary.json", {"config": {}})
    written = report("error_detection", s)
    assert [p.suffix for p in written] == [".md"]


def test_report_unknown_and_missing(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        report("no_such", settings_for(tmp_path))
    with pytest.raises(FileNotFoundError):
        report("benchmarks", settings_for(tmp_path))


def test_tables_file_content(tmp_path: Path) -> None:
    s = settings_for(tmp_path)
    write_outputs(s)
    (path,) = tables.TABLES["knowledge_boundary"](s)
    text = path.read_text()
    assert text.startswith("# Knowledge boundary")
    assert json.loads((s.out("knowledge_boundary") / "summary.json").read_text())
