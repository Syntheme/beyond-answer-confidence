"""Static figures drawn from experiment outputs.

Every plotted value is also in the experiment's ``summary.json`` or
``rows.jsonl`` and in the tables of :mod:`beyond_answer_confidence.reporting.tables`,
so a figure is never the only source. Shaded bands and error bars are 95 %
intervals over items: Wilson intervals for accuracies, mean +/- 1.96
standard errors for mean probabilities.

Matplotlib is imported only when a figure is drawn (with the ``Agg``
backend, so no display is needed).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from beyond_answer_confidence.experiments.base import read_jsonl
from beyond_answer_confidence.metrics.calibration import (
    normal_mean_interval,
    wilson_interval,
)
from beyond_answer_confidence.settings import Settings

# Categorical slots 1-3 (validated for colour-vision deficiency on the light
# surface; the third has low contrast, so every series also has a marker and
# a legend entry, and the tables repeat the values).
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
IDEAL = "#8a8984"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
Z = 1.96

Bins = list[dict[str, float]]
"""Reliability bins: ``{"confidence", "accuracy", "n"}`` per bin."""


def ci_mean(values: Any) -> tuple[float, float]:
    """95 % normal interval for a mean.

    Args:
        values: One value per item.

    Returns:
        ``(low, high)``.
    """
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return normal_mean_interval(v, Z)


def ci_prop(values: Any) -> tuple[float, float]:
    """95 % Wilson interval for a proportion.

    Args:
        values: One boolean per item.

    Returns:
        ``(low, high)`` (NaN for no items).
    """
    v = np.asarray(values, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return wilson_interval(float(v.mean()), len(v), Z)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _band(ax: Any, xs: Any, bounds: Any, color: str) -> None:
    lo, hi = zip(*bounds, strict=True)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)


def _bars(ax: Any, xs: Any, ys: Any, bounds: Any, color: str) -> None:
    if not len(bounds):
        return
    lo, hi = (np.array(b, float) for b in zip(*bounds, strict=True))
    y = np.asarray(ys, float)
    ax.errorbar(
        xs,
        y,
        yerr=[y - lo, hi - y],
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=2.5,
    )


def _line(ax: Any, xs: Any, ys: Any, color: str, label: str, marker: str = "o") -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        lw=2,
        marker=marker,
        ms=7,
        mec=SURFACE,
        mew=1.5,
        label=label,
    )


def _style(ax: Any, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)


def _legend(ax: Any, loc: str = "lower right") -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc=loc, labelcolor=TEXT)


def _save(fig: Any, out: Path, dpi: int = 150) -> Path:
    plt = _plt()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out


def _binned_reliability(
    ax: Any,
    conf: np.ndarray,
    correct: np.ndarray,
    edges: np.ndarray,
    min_n: int,
    color: str,
    label: str,
    marker: str,
) -> None:
    """Accuracy against mean confidence in fixed-width bins with enough items."""
    idx = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
    xs, ys, bounds = [], [], []
    for bi in range(len(edges) - 1):
        m = idx == bi
        if m.sum() >= min_n:
            xs.append(float(conf[m].mean()))
            ys.append(float(correct[m].mean()))
            bounds.append(ci_prop(correct[m]))
    _bars(ax, xs, ys, bounds, color)
    _line(ax, xs, ys, color, label, marker=marker)


# --- synthetic worlds -------------------------------------------------------

PROFILE_LABELS = {
    "uniform": "uniform\n25 % each",
    "skewed": "skewed\n40/30/20/10",
    "peaked": "peaked\n70/10/10/10",
    "binary": "two-way\n50/50",
}
SETTLED_GROUPS = (
    ("decided,\nnot\ngiven", "D0", True),
    ("decided,\nand\ngiven", "D3", True),
    ("drawn,\nresult\nsealed", "SP", True),
    ("future,\nodds\ngiven", "SFp", False),
    ("future,\nodds\nunknown", "US", False),
)


def synthetic_worlds_figure(
    rows: Sequence[Mapping[str, Any]], out: Path, seed: int = 0
) -> Path:
    """Three panels: stated chance, missing knowledge, the ``settled`` question.

    Args:
        rows: Synthetic-worlds rows.
        out: PNG path.
        seed: Scenario seed of the run (to recover the stated chances).

    Returns:
        ``out``.
    """
    from beyond_answer_confidence.tasks.synthetic import PROFILES, make_scenario

    plt = _plt()
    main = [r for r in rows if not r["alone"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)

    ax = axes[0]
    ticks: list[float] = []
    labels: list[str] = []
    for j, prof in enumerate(PROFILES):
        g = [r for r in main if r["cell"] == "SFp" and r["profile"] == prof]
        if not g:
            continue
        ranked = np.zeros((len(g), 4))
        for i, r in enumerate(g):
            sc = make_scenario(r["scenario"], seed)
            listed = sorted(sc.p_star, key=lambda o: -sc.p_star[o])
            ranked[i] = [r["dist"][o] for o in listed]
        stated = sorted((c / 20 for c in PROFILES[prof]), reverse=True)
        xs = j * 5 + np.arange(4)
        ax.bar(
            xs - 0.2,
            stated,
            0.36,
            color=SURFACE,
            edgecolor=IDEAL,
            lw=1.2,
            label="stated chance" if not ticks else None,
        )
        means = ranked.mean(axis=0)
        ax.bar(
            xs + 0.2,
            means,
            0.36,
            color=S1,
            label="mean probability" if not ticks else None,
        )
        _bars(ax, xs + 0.2, means, [ci_mean(c) for c in ranked.T], TEXT)
        ticks.append(j * 5 + 1.5)
        labels.append(PROFILE_LABELS.get(prof, prof))
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("probability (options by stated chance)", color=TEXT_2, fontsize=8)
    _legend(ax, "upper center")
    _style(ax, "a. Stated chance of a future draw")

    ax = axes[1]
    doses = ["D0", "D1", "D2", "D3"]
    ideal = [0.25, 1 / 3, 0.5, 1.0]
    vals = [[r["p_max"] for r in main if r["cell"] == c] for c in doses]
    _band(ax, range(4), [ci_mean(v) for v in vals], S1)
    ax.plot(
        range(4),
        ideal,
        color=IDEAL,
        lw=1.2,
        ls="--",
        marker="o",
        ms=6,
        label="ideal: 1 / options left",
    )
    _line(ax, range(4), [np.mean(v) if v else np.nan for v in vals], S1, "observed")
    ax.set_xticks(range(4), ["nothing", "1 ruled out", "2 ruled out", "answer given"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean top probability", color=TEXT_2, fontsize=9)
    _legend(ax, "upper left")
    _style(ax, "b. A fixed fact, partly or fully known")

    ax = axes[2]
    for i, (_, cell, settled) in enumerate(SETTLED_GROUPS):
        v = [r["p_settled"] for r in main if r["cell"] == cell and "p_settled" in r]
        if not v:
            continue
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(
            np.full(len(v), i) + jitter,
            v,
            s=8,
            color=S1 if settled else S2,
            alpha=0.35,
            lw=0,
        )
        ax.plot([i - 0.3, i + 0.3], [np.mean(v)] * 2, color=TEXT, lw=2)
    ax.set_xticks(
        range(len(SETTLED_GROUPS)), [g[0] for g in SETTLED_GROUPS], fontsize=8
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('P(yes) to "is the answer already fixed?"', color=TEXT_2, fontsize=9)
    ax.text(
        0.03,
        0.5,
        "blue: answer fixed (ideal yes)\norange: not yet decided (ideal no)\n"
        "black bar: mean",
        transform=ax.transAxes,
        fontsize=8,
        color=TEXT_2,
        va="center",
    )
    _style(ax, "c. Is the answer settled?")
    return _save(fig, out)


# --- knowledge boundary -----------------------------------------------------


def _popularity_panel(ax: Any, df: Any, res: Mapping[str, Any]) -> None:
    import pandas as pd

    q = res["popqa_by_quintile"]
    keys = sorted(q, key=int)
    xs = np.arange(len(keys))
    real = df[df["set"] == "popqa"].copy()
    real["quintile"] = pd.qcut(real["s_pop"].rank(method="first"), 5, labels=False)
    groups = [real[real["quintile"] == int(k)] for k in keys]
    _band(ax, xs, [ci_prop(g["correct"]) for g in groups], S2)
    _band(ax, xs, [ci_mean(g["p_max"]) for g in groups], S1)
    _band(ax, xs, [ci_mean(g["p_known"]) for g in groups], S3)
    _line(ax, xs, [q[k]["accuracy"] for k in keys], S2, "accuracy")
    _line(ax, xs, [q[k]["mean_p_max"] for k in keys], S1, "mean top probability")
    _line(
        ax,
        xs,
        [q[k]["mean_p_known"] for k in keys],
        S3,
        'P(yes) "do you know?"',
        marker="s",
    )
    fab = res["fabricated"]
    made_up = df[df["set"] == "fabricated"]
    for key, col, color in (
        ("mean_p_max", "p_max", S1),
        ("mean_p_known", "p_known", S3),
    ):
        _bars(ax, [-1.2], [fab[key]], [ci_mean(made_up[col])], color)
        ax.scatter(
            [-1.2],
            [fab[key]],
            color=color,
            s=60,
            marker="D",
            edgecolor=SURFACE,
            zorder=3,
        )
    ax.axhline(0.25, color=IDEAL, lw=1, ls="--")
    ax.text(
        len(keys) - 0.7, 0.265, "chance (1 in 4)", color=TEXT_2, fontsize=8, ha="right"
    )
    labels = ["made-up\nentities"] + [
        f"{q[k]['s_pop_range'][0]:,}-\n{q[k]['s_pop_range'][1]:,}" for k in keys
    ]
    ax.set_xticks([-1.2, *xs], labels, fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(
        "subject popularity quintile (monthly page views)", color=TEXT_2, fontsize=9
    )
    _legend(ax)
    _style(ax, "a. PopQA by subject popularity")


def _news_panel(ax: Any, df: Any, res: Mapping[str, Any]) -> None:
    import pandas as pd

    tf = df[df["set"] == "oracle_tf"].copy()
    tf["q"] = pd.PeriodIndex(tf["month"], freq="M").asfreq("Q").astype(str)
    g = tf.groupby("q").agg(
        acc=("correct", "mean"), pm=("p_max", "mean"), kn=("p_known", "mean")
    )
    xs = np.arange(len(g))
    by_q = [tf[tf["q"] == qq] for qq in g.index]
    _band(ax, xs, [ci_prop(x["correct"]) for x in by_q], S2)
    _band(ax, xs, [ci_mean(x["p_max"]) for x in by_q], S1)
    _band(ax, xs, [ci_mean(x["p_known"]) for x in by_q], S3)
    _line(ax, xs, g["acc"], S2, "accuracy (yes/no)")
    _line(ax, xs, g["pm"], S1, "mean top probability")
    _line(ax, xs, g["kn"], S3, 'P(yes) "do you know how it turned out?"', marker="s")
    cp = res.get("oracle_change_point") or {}
    # The post-break comparisons exist only when the drop was large enough.
    if "calibration_after_cutoff" in res and cp.get("first_post_month"):
        period = pd.Period(cp["first_post_month"], freq="M")
        quarters = list(g.index)
        if str(period.asfreq("Q")) in quarters:
            i = quarters.index(str(period.asfreq("Q")))
            vx = i - 0.5 + ((period.month - 1) % 3) / 3
            ax.axvline(vx, color=IDEAL, lw=1.2, ls="--")
            ax.text(
                vx - 0.1,
                0.97,
                f"accuracy break, {cp['first_post_month']} (month uncertain)",
                color=TEXT_2,
                fontsize=8,
                ha="right",
                va="top",
            )
    ax.axhline(0.5, color=IDEAL, lw=0.8, ls=":")
    step = max(1, len(g) // 9)
    ax.set_xticks(xs[::step], list(g.index)[::step], fontsize=8)
    ax.set_ylim(0, 1.02)
    _legend(ax, "lower left")
    _style(ax, "b. Daily Oracle: yes/no news questions by quarter")


def knowledge_boundary_figure(
    rows: Sequence[Mapping[str, Any]], results: Mapping[str, Any], out: Path
) -> Path:
    """PopQA by popularity (with fabricated entities) and news by quarter.

    Args:
        rows: Knowledge-boundary rows.
        results: The ``results`` block of its summary.
        out: PNG path.

    Returns:
        ``out``.
    """
    import pandas as pd

    plt = _plt()
    df = pd.DataFrame(list(rows))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.2),
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1.6]},
    )
    _popularity_panel(axes[0], df, results)
    _news_panel(axes[1], df, results)
    return _save(fig, out)


# --- evidence sufficiency ---------------------------------------------------

HOTPOT_CELLS = ("closed", "dose0", "dose1", "dose2")
HOTPOT_LABELS = (
    "question\nonly",
    "2 paragraphs,\nnot needed",
    "1 of 2\nneeded",
    "both\nneeded",
)


def evidence_sufficiency_figure(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    out: Path,
    min_n: int = 100,
) -> Path:
    """HotpotQA evidence cells and Quizbowl clue by clue.

    Args:
        rows: Evidence-sufficiency rows (for the intervals).
        results: The ``results`` block of its summary.
        out: PNG path.
        min_n: Smallest number of questions for a Quizbowl point.

    Returns:
        ``out``.
    """
    import pandas as pd

    plt = _plt()
    df = pd.DataFrame(list(rows))
    hp, qb = df[df["set"] == "hotpot"], df[df["set"] == "quizbowl"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), facecolor=SURFACE)
    ax = axes[0]
    h = results["hotpot_by_cell"]
    cells = [c for c in HOTPOT_CELLS if c in h]
    xs = np.arange(len(cells))
    hg = [hp[hp["cell"] == c] for c in cells]
    _band(ax, xs, [ci_prop(g["correct"]) for g in hg], S2)
    _band(ax, xs, [ci_mean(g["p_max"]) for g in hg], S1)
    _band(ax, xs, [ci_mean(g["p_enough"]) for g in hg], S3)
    _line(ax, xs, [h[c]["accuracy"] for c in cells], S2, "accuracy")
    _line(ax, xs, [h[c]["mean_p_max"] for c in cells], S1, "mean top probability")
    _line(
        ax,
        xs,
        [h[c]["mean_p_enough"] for c in cells],
        S3,
        'P(yes) "enough information?"',
        marker="s",
    )
    names = dict(zip(HOTPOT_CELLS, HOTPOT_LABELS, strict=True))
    ax.set_xticks(xs, [names[c] for c in cells], fontsize=8)
    ax.set_ylim(0, 1.02)
    _legend(ax)
    _style(ax, "a. HotpotQA comparisons: evidence given")

    ax = axes[1]
    k = results["quizbowl_by_k"]
    ks = [x for x in sorted(k, key=int) if k[x]["n"] >= min_n]
    kx = [int(x) for x in ks]
    kg = [qb[qb["k"] == int(x)] for x in ks]
    _band(ax, kx, [ci_prop(g["correct"]) for g in kg], S2)
    _band(ax, kx, [ci_mean(g["p_max"]) for g in kg], S1)
    _band(ax, kx, [ci_mean(g["p_enough"]) for g in kg], S3)
    _line(ax, kx, [k[x]["accuracy"] for x in ks], S2, "accuracy")
    _line(ax, kx, [k[x]["mean_p_max"] for x in ks], S1, "mean top probability")
    _line(
        ax,
        kx,
        [k[x]["mean_p_enough"] for x in ks],
        S3,
        'P(yes) "clues identify it?"',
        marker="s",
    )
    ax.set_xlabel(
        f"sentences of the question shown (k with >= {min_n} questions)",
        color=TEXT_2,
        fontsize=9,
    )
    ax.set_ylim(0, 1.02)
    _legend(ax)
    _style(ax, "b. Quizbowl: clues revealed one at a time")
    return _save(fig, out)


# --- benchmarks and option count ---------------------------------------------


def _calibration_line(ax: Any, lo: float) -> None:
    ax.plot([0, 1], [0, 1], color=IDEAL, lw=1.2, ls="--", label="perfect calibration")
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy", color=TEXT_2, fontsize=9)


def benchmarks_figure(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    """Reliability on factual QA and uncertainty against human disagreement.

    Args:
        rows: Benchmark rows.
        out: PNG path.

    Returns:
        ``out``.
    """
    import pandas as pd

    plt = _plt()
    df = pd.DataFrame(list(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    ax = axes[0]
    edges = np.linspace(0.25, 1.0, 7)
    for name, color, label, marker in (
        ("triviaqa", S1, "TriviaQA", "o"),
        ("simpleqa", S2, "SimpleQA Verified", "s"),
        ("truthfulqa_binary", S3, "TruthfulQA (2 options)", "^"),
    ):
        g = df[df["set"] == name]
        _binned_reliability(
            ax,
            g["p_max"].to_numpy(float),
            g["correct"].to_numpy(float),
            edges,
            20,
            color,
            label,
            marker,
        )
    _calibration_line(ax, 0.2)
    ax.set_xlabel(
        "top probability (bins with >= 20 items; bars: 95 % interval)",
        color=TEXT_2,
        fontsize=9,
    )
    _legend(ax, "upper left")
    _style(ax, "a. Factual QA: confidence against accuracy")

    ax = axes[1]
    nli = df[df["set"] == "chaos_nli"].copy()
    if len(nli):
        nli["h"] = nli["human_entropy"] / np.log2(3)
        edges = np.quantile(nli["h"], np.linspace(0, 1, 9))
        nli["bin"] = np.clip(np.digitize(nli["h"], edges[1:-1]), 0, 7)
        g = nli.groupby("bin").agg(h=("h", "mean"), j=("norm_entropy", "mean"))
        _bars(
            ax,
            g["h"],
            g["j"],
            [ci_mean(nli[nli["bin"] == b]["norm_entropy"]) for b in g.index],
            S1,
        )
        _line(ax, g["h"], g["j"], S1, "mean per human-entropy octile")
    ax.plot([0, 1], [0, 1], color=IDEAL, lw=1.2, ls="--", label="equal to humans")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(
        "human label entropy (100 annotators, normalised)", color=TEXT_2, fontsize=9
    )
    ax.set_ylabel("normalised entropy of the probabilities", color=TEXT_2, fontsize=9)
    _legend(ax, "upper left")
    _style(ax, "b. ChaosNLI: hesitation where humans disagree")
    return _save(fig, out)


def option_count_figure(rows: Sequence[Mapping[str, Any]], out: Path) -> Path:
    """Reliability on standard multiple choice, and intents against option count.

    Args:
        rows: Option-count rows (``set``, ``p_max``, ``correct``; ``error_type``
            for MMLU-Redux and ``K`` for the intent sets).
        out: PNG path.

    Returns:
        ``out``.
    """
    import pandas as pd

    plt = _plt()
    df = pd.DataFrame(list(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    ax = axes[0]
    edges = np.linspace(0.3, 1.0, 8)
    for name, color, label, marker in (
        ("mmlu_redux", S1, "MMLU-Redux (error-free items)", "o"),
        ("mmlu_cf", S2, "MMLU-CF", "s"),
        ("anli", S3, "ANLI", "^"),
    ):
        g = df[df["set"] == name]
        if name == "mmlu_redux" and "error_type" in g:
            g = g[g["error_type"] == "ok"]
        _binned_reliability(
            ax,
            g["p_max"].to_numpy(float),
            g["correct"].to_numpy(float),
            edges,
            30,
            color,
            label,
            marker,
        )
    _calibration_line(ax, 0.3)
    ax.set_xlabel(
        "top probability (bins with >= 30 items; bars: 95 % interval)",
        color=TEXT_2,
        fontsize=9,
    )
    _legend(ax, "upper left")
    _style(ax, "a. Standard multiple choice")

    ax = axes[1]
    clinc = df[df["set"] == "clinc_k"]
    if len(clinc):
        c = clinc.groupby("K").agg(acc=("correct", "mean"), pm=("p_max", "mean"))
        pos = np.arange(len(c))
        ck = [clinc[clinc["K"] == k] for k in c.index]
        _band(ax, pos, [ci_prop(x["correct"]) for x in ck], S2)
        _band(ax, pos, [ci_mean(x["p_max"]) for x in ck], S1)
        _line(ax, pos, c["acc"], S2, "accuracy")
        _line(ax, pos, c["pm"], S1, "mean top probability", marker="s")
        ax.set_xticks(pos, [str(int(k)) for k in c.index])
        low = float(min(c["acc"].min(), c["pm"].min()))
        ax.set_ylim(max(0.0, math.floor((low - 0.05) * 10) / 10), 1.01)
    ax.set_xlabel("number of options (CLINC150 intents)", color=TEXT_2, fontsize=9)
    _legend(ax, "lower left")
    _style(ax, "b. Accuracy and confidence by number of options")
    return _save(fig, out)


# --- reliability panels -----------------------------------------------------


def _segment_distance(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Distance from point ``p`` to the segment ``a``-``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = (
        0.0
        if length2 == 0
        else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    )
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def label_anchor(
    points: Sequence[tuple[float, float]],
    others: Sequence[Sequence[tuple[float, float]]],
) -> tuple[float, float]:
    """Pick the point of a series to label: far from other lines and edges.

    Args:
        points: The series' ``(x, y)`` points.
        others: The other series' polylines in the panel.

    Returns:
        The chosen ``(x, y)``.
    """

    def clearance(pt: tuple[float, float]) -> float:
        dists = [
            _segment_distance(pt, line[i], line[min(i + 1, len(line) - 1)])
            for line in others
            for i in range(len(line))
        ]
        return min(dists, default=1.0)

    def score(pt: tuple[float, float]) -> float:
        edge = 0.3 if (pt[1] > 0.95 or pt[0] < 0.05 or pt[0] > 0.97) else 0.0
        return clearance(pt) - edge

    return max(points, key=score)


def _interval_bars(
    ax: Any, pts: Bins, x: list[float], y: list[float], color: str
) -> None:
    """Draw 95 % Wilson bars when every bin carries its item count."""
    if not all("n" in p for p in pts):
        return
    bounds = [wilson_interval(p["accuracy"], int(p["n"]), Z) for p in pts]
    _bars(ax, x, y, bounds, color)


def _direct_label(
    ax: Any,
    series: Mapping[str, Bins],
    key: str,
    x: list[float],
    y: list[float],
    text: str,
) -> None:
    others = [
        [(p["confidence"], p["accuracy"]) for p in po]
        for k, po in series.items()
        if k != key and po
    ]
    lx, ly = label_anchor(list(zip(x, y, strict=True)), others)
    right = lx > 0.6
    flat = [pt for line in others for pt in line]
    near = min(flat, key=lambda o: abs(o[0] - lx), default=(lx, -1.0))
    dy = 10 if ly >= near[1] or ly < 0.15 else -16
    ax.annotate(
        text,
        (lx, ly),
        xytext=(-8 if right else 8, dy),
        textcoords="offset points",
        ha="right" if right else "left",
        color=TEXT,
        fontsize=8,
    )


def plot_reliability_panels(
    panels: Sequence[tuple[str, Mapping[str, Bins]]],
    styles: Mapping[str, tuple[str, str, str]],
    title: str,
    path: Path,
    *,
    ncols: int = 3,
    direct_labels: bool = False,
    short_names: Mapping[str, str] | None = None,
) -> Path:
    """Small-multiple reliability diagrams.

    Args:
        panels: ``(panel title, {series: bins})`` per panel.
        styles: Series -> ``(display name, colour, marker)``.
        title: Figure title.
        path: Output PNG.
        ncols: Panels per row.
        direct_labels: Label each line once per panel.
        short_names: Shorter names for direct labels (the legend keeps the
            full names).

    Returns:
        ``path``.
    """
    plt = _plt()
    short = dict(short_names or {})
    ncols = max(1, min(ncols, len(panels)))
    nrows = max(1, math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        sharex=True,
        sharey=True,
        facecolor=SURFACE,
        squeeze=False,
    )
    for ax in axes.flat[len(panels) :]:
        ax.set_visible(False)
    for ax, (panel_title, series) in zip(axes.flat, panels, strict=False):
        ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1, ls=(0, (3, 3)), zorder=1)
        for key, (name, color, marker) in styles.items():
            pts = series.get(key)
            if not pts:
                continue
            x = [p["confidence"] for p in pts]
            y = [p["accuracy"] for p in pts]
            ax.plot(
                x, y, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2,
                zorder=3, label=name,
            )  # fmt: skip
            _interval_bars(ax, pts, x, y, color)
            if direct_labels:
                _direct_label(ax, series, key, x, y, short.get(key, name))
        _style(ax, panel_title)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("mean top-label probability (bin)", color=TEXT_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (bin)", color=TEXT_2, fontsize=9)
    seen: dict[str, Any] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen),
        loc="upper right",
        frameon=False,
        labelcolor=TEXT,
        fontsize=10,
    )
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrows == 1 else 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


CONDITION_TITLES = {
    "L0": "no information",
    "L1": "1 example per intent",
    "L2": "2 examples per intent",
    "L4": "4 examples per intent",
    "L8": "8 examples per intent",
    "Lname": "intent names",
    "Lname+8": "names + 8 examples",
    "names": "names (classifier only)",
}
"""Readable panel titles for the knowledge conditions."""

DATASET_STYLES = {
    "banking77": ("Banking77", S1, "o"),
    "clinc150": ("CLINC150", S2, "s"),
}


def intent_reliability_figure(
    results: Mapping[str, Mapping[str, Any]], out: Path
) -> Path:
    """Reliability per knowledge condition, one series per intent dataset.

    Args:
        results: Error-detection results keyed by dataset (each with
            ``reliability_bins`` keyed by condition).
        out: PNG path.

    Returns:
        ``out``.
    """
    from beyond_answer_confidence.metrics.calibration import merge_tied_bins

    conds: list[str] = []
    for res in results.values():
        conds += [c for c in res.get("reliability_bins", {}) if c not in conds]
    panels = [
        (
            CONDITION_TITLES.get(c, c),
            {
                name: merge_tied_bins([dict(b) for b in res["reliability_bins"][c]])
                for name, res in results.items()
                if c in res.get("reliability_bins", {})
            },
        )
        for c in conds
    ]
    styles = {k: v for k, v in DATASET_STYLES.items() if k in results}
    for name in results:
        styles.setdefault(name, (name, S3, "^"))
    return plot_reliability_panels(
        panels,
        styles,
        "Confidence against accuracy, intent test sets (tied bins merged; 95 % intervals)",
        out,
        direct_labels=True,
    )


def comparator_reliability_figure(
    bins: Mapping[str, Mapping[str, Bins]], out: Path, dataset: str
) -> Path:
    """Reliability of the API and a local classifier on the same items.

    Args:
        bins: ``{condition: {"jev": bins, "gliner": bins}}``.
        out: PNG path.
        dataset: Dataset name for the title.

    Returns:
        ``out``.
    """
    panels = [
        (CONDITION_TITLES.get(c, c), {k: v for k, v in series.items() if v})
        for c, series in bins.items()
        if series.get("gliner")
    ]
    styles = {
        "jev": ("API", S1, "o"),
        "gliner": ("GLiNER2.5-Decide", S2, "s"),
    }
    return plot_reliability_panels(
        panels,
        styles,
        f"API vs local classifier reliability, {dataset}, same items "
        "(equal-mass bins, 95 % intervals)",
        out,
        ncols=len(panels) or 1,
        direct_labels=True,
        short_names={"gliner": "GLiNER"},
    )


# --- registry ---------------------------------------------------------------


def _summary(settings: Settings, name: str) -> dict[str, Any]:
    import json

    path = settings.out(name) / "summary.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _rows(settings: Settings, name: str) -> list[dict[str, Any]]:
    return read_jsonl(settings.out(name) / "rows.jsonl")


def _fig_path(settings: Settings, name: str, stem: str | None = None) -> Path:
    return settings.out(name) / "figures" / f"{stem or name}.png"


def _report_synthetic(settings: Settings) -> list[Path]:
    seed = int(_summary(settings, "synthetic_worlds").get("config", {}).get("seed", 0))
    rows = _rows(settings, "synthetic_worlds")
    return [
        synthetic_worlds_figure(rows, _fig_path(settings, "synthetic_worlds"), seed)
    ]


def _report_boundary(settings: Settings) -> list[Path]:
    res = _summary(settings, "knowledge_boundary").get("results")
    if res is None:
        return []
    rows = _rows(settings, "knowledge_boundary")
    return [
        knowledge_boundary_figure(rows, res, _fig_path(settings, "knowledge_boundary"))
    ]


def _report_evidence(settings: Settings) -> list[Path]:
    res = _summary(settings, "evidence_sufficiency").get("results")
    if res is None:
        return []
    rows = _rows(settings, "evidence_sufficiency")
    return [
        evidence_sufficiency_figure(
            rows, res, _fig_path(settings, "evidence_sufficiency")
        )
    ]


def _report_benchmarks(settings: Settings) -> list[Path]:
    return [
        benchmarks_figure(
            _rows(settings, "benchmarks"), _fig_path(settings, "benchmarks")
        )
    ]


def _report_options(settings: Settings) -> list[Path]:
    rows = _rows(settings, "option_count")
    return [option_count_figure(rows, _fig_path(settings, "option_count"))]


def _dataset_blocks(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        k: v
        for k, v in summary.items()
        if isinstance(v, Mapping) and "reliability_bins" in v
    }


def _report_errors(settings: Settings) -> list[Path]:
    blocks = _dataset_blocks(_summary(settings, "error_detection"))
    if not blocks:
        return []
    return [intent_reliability_figure(blocks, _fig_path(settings, "error_detection"))]


def _report_comparator(settings: Settings) -> list[Path]:
    # The comparison writes its summary under the classifier's directory.
    blocks = _dataset_blocks(_summary(settings, "gliner"))
    return [
        comparator_reliability_figure(
            res["reliability_bins"],
            _fig_path(settings, "gliner", f"gliner_comparison_{name}"),
            name,
        )
        for name, res in blocks.items()
    ]


FIGURES: dict[str, Callable[[Settings], list[Path]]] = {
    "synthetic_worlds": _report_synthetic,
    "knowledge_boundary": _report_boundary,
    "evidence_sufficiency": _report_evidence,
    "benchmarks": _report_benchmarks,
    "option_count": _report_options,
    "error_detection": _report_errors,
    "gliner_comparison": _report_comparator,
}
"""Experiment or analysis name -> draws its figures from the outputs."""
