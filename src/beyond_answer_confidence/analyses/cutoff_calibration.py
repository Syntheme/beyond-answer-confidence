"""Calibration across the knowledge cutoff (offline; reads saved rows).

On scored dated yes/no news rows only (no requests):

1. **Accuracy at equal confidence, before vs after the cutoff.** At equal
   ``p_max``, are later questions less often right? Accuracy per ``p_max``
   band for each period with a month-cluster bootstrap (months resampled
   within each period), and the pooled band-wise map of ``p_max`` alone
   (each band set to its pooled accuracy, fitted in-sample on both periods;
   coarse and fine bands) with the calibration gap it still leaves in each
   period. No map that depends on ``p_max`` alone can be calibrated on both
   sides if accuracy differs at equal confidence.
2. **The "not known" option at matched coverage, as the share of post-cutoff
   questions varies.** Uses the balanced cutoff sample of the follow-up
   questions experiment (equal numbers on each side). Reweighting the same
   items gives the option's accuracy and that of confidence and P(known)
   thresholds at exactly the option's answer rate (the boundary item is
   split fractionally and tied scores share it evenly; achieved coverage is
   reported) for any post-cutoff share from 0 to 1. Intervals at selected
   shares come from a period-stratified moving-block bootstrap over months
   (6-month blocks), conditional on the observed three-call answers. One
   reference share is the unreweighted share in the full dated yes/no panel
   (1,680 of 6,320), a property of this dated sample, not a deployment
   prevalence.

Inputs: ``<output_dir>/knowledge_boundary/rows.jsonl`` (set ``oracle_tf``)
and ``<output_dir>/follow_up_questions/rows.jsonl`` (units ``cutoff:*``).
Output: ``<output_dir>/cutoff_calibration/summary.json`` and, optionally,
``figures/cutoff_calibration.png`` in the same directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.rows import experiment_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, write_json
from beyond_answer_confidence.metrics.ranking import weighted_accuracy_at
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import month_block_indices, percentile_ci
from beyond_answer_confidence.tasks.follow_up import CUTOFF_MONTH

__all__ = [
    "BAND_LABELS",
    "BLOCK_MONTHS",
    "COARSE",
    "CUTOFF_MONTH",
    "FINE",
    "PANEL_SHARE",
    "REPORT_SHARES",
    "SHARES",
    "at_share",
    "band_table",
    "best_single_map_gaps",
    "cutoff_rows",
    "equal_confidence",
    "figure",
    "news_rows",
    "paired_rows",
    "prevalence",
]

NAME = "cutoff_calibration"
COARSE = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
"""Coarse ``p_max`` band edges (the last edge includes 1.0)."""
FINE = (*np.round(np.arange(0.5, 1.0, 0.025), 3).tolist(), 1.0001)
"""Fine ``p_max`` band edges, 0.025 wide."""
PANEL_SHARE = 1680 / 6320
"""Unreweighted post-cutoff share of the full dated yes/no panel."""
BLOCK_MONTHS = 6
"""Block length of the month bootstrap."""
SHARES = tuple(np.round(np.arange(0, 1.0001, 0.05), 2).tolist())
"""Post-cutoff shares of the prevalence curve."""
REPORT_SHARES = (0.05, 0.1, 0.2, PANEL_SHARE, 0.5, 0.8)
"""Shares with bootstrap intervals."""
BAND_LABELS = ("0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0")
"""Labels of the coarse bands."""
NEWS_SET = "oracle_tf"
"""Row ``set`` of the dated yes/no news questions."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples (both analyses share one generator,
            equal-confidence first).
        seed: Bootstrap seed.
        figures: Also draw the figure into the output directory.
        precise_float: Parse row floats round-trip exactly instead of with
            the default parser (kept for reproducibility; changes tie
            groups of ``p_max`` and ``p_known``; see
            :mod:`beyond_answer_confidence.analyses.rows`).
    """

    bootstrap: int = 2000
    seed: int = 0
    figures: bool = True
    precise_float: bool = False


# --- 1. accuracy at equal confidence ------------------------------------------


def band_table(tf: pd.DataFrame, edges: tuple[float, ...]) -> pd.DataFrame:
    """Items, mean confidence and accuracy per band and period.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct`` and ``post``.
        edges: Band edges on ``p_max`` (left-closed bands).

    Returns:
        One row per band: ``band``, then ``n``, ``acc`` and ``conf`` for each
        period, suffixed ``_pre`` and ``_post``.
    """
    d = tf.assign(band=pd.cut(tf["p_max"], list(edges), right=False))
    g = d.groupby(["band", "post"], observed=False).agg(
        n=("correct", "size"), acc=("correct", "mean"), conf=("p_max", "mean")
    )
    out = cast(pd.DataFrame, g.unstack("post"))
    pairs = cast(list[tuple[str, bool]], list(out.columns))
    out.columns = [f"{a}_{'post' if b else 'pre'}" for a, b in pairs]
    return out.reset_index()


def best_single_map_gaps(
    tf: pd.DataFrame, edges: tuple[float, ...]
) -> dict[str, float]:
    """Calibration gap left by the most favourable band-wise map of ``p_max``.

    Each band is mapped to its pooled accuracy over both periods (fitted
    in-sample, so this favours the map). Any function of ``p_max`` constant
    on bands cannot do better on the pooled data; the gaps it leaves within
    each period show what no such map can remove.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct`` and ``post``.
        edges: Band edges.

    Returns:
        Mean (recalibrated confidence - accuracy) per period, and the mean
        absolute band-level gap per period weighted by items.
    """
    d = tf.assign(band=pd.cut(tf["p_max"], list(edges), right=False))
    pooled = d.groupby("band", observed=False)["correct"].transform("mean")
    d = d.assign(mapped=pooled.astype(float), ok=d["correct"].astype(float))
    out: dict[str, float] = {}
    for name, g in (("pre", d[~d["post"]]), ("post", d[d["post"]])):
        out[f"mean_gap_{name}"] = float((g["mapped"] - g["ok"]).mean())
        per_band = g.groupby("band", observed=True).agg(
            m=("mapped", "mean"), a=("ok", "mean"), n=("ok", "size")
        )
        out[f"weighted_abs_band_gap_{name}"] = float(
            (np.abs(per_band["m"] - per_band["a"]) * per_band["n"]).sum()
            / per_band["n"].sum()
        )
    return out


def _ci_of(mat: np.ndarray, i: int) -> tuple[float, float] | None:
    col = mat[:, i]
    col = col[~np.isnan(col)]
    return percentile_ci(col) if len(col) else None


def equal_confidence(
    tf: pd.DataFrame, b: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Pre vs post accuracy at equal confidence, with month-cluster intervals.

    Months are resampled with replacement within each period (pre first,
    then post, in every resample), so both periods keep their number of
    months. The cutoff is fixed; see the knowledge-cutoff analysis for
    intervals that repeat its selection.

    Args:
        tf: Dated yes/no rows with ``p_max``, ``correct``, ``post`` and
            ``month``.
        b: Bootstrap resamples.
        rng: Random generator.

    Returns:
        Per coarse band: counts, accuracies, difference and 95 % intervals;
        the best band-wise map's gaps for coarse and fine bands, with a 95 %
        interval for the post-cutoff mean gap.
    """
    months = np.array(sorted(tf["month"].unique()))
    by_month = dict(tuple(tf.groupby("month")))
    post_months = np.array([m for m in months if bool(by_month[m]["post"].iloc[0])])
    pre_months = np.array([m for m in months if not bool(by_month[m]["post"].iloc[0])])
    table = band_table(tf, COARSE)
    diffs, gaps_c, gaps_f, acc_pre, acc_post = [], [], [], [], []
    for _ in range(b):
        pick = [
            *rng.choice(pre_months, len(pre_months)),
            *rng.choice(post_months, len(post_months)),
        ]
        r = pd.concat([by_month[m] for m in pick])
        t = band_table(r, COARSE)
        diffs.append((t["acc_pre"] - t["acc_post"]).to_numpy(float))
        acc_pre.append(t["acc_pre"].to_numpy(float))
        acc_post.append(t["acc_post"].to_numpy(float))
        gaps_c.append(best_single_map_gaps(r, COARSE)["mean_gap_post"])
        gaps_f.append(best_single_map_gaps(r, FINE)["mean_gap_post"])
    dmat, pre_m, post_m = np.array(diffs), np.array(acc_pre), np.array(acc_post)
    bands = [
        {
            "band": str(row["band"]),
            "n_pre": int(row["n_pre"]),
            "n_post": int(row["n_post"]),
            "accuracy_pre": float(row["acc_pre"]),
            "accuracy_post": float(row["acc_post"]),
            "accuracy_pre_ci95": _ci_of(pre_m, i),
            "accuracy_post_ci95": _ci_of(post_m, i),
            "difference": float(row["acc_pre"] - row["acc_post"]),
            "difference_ci95": _ci_of(dmat, i),
        }
        for i, (_, row) in enumerate(table.iterrows())
    ]
    coarse, fine = best_single_map_gaps(tf, COARSE), best_single_map_gaps(tf, FINE)
    return {
        "bands": bands,
        "best_single_map": {
            "coarse_bands": {
                **coarse,
                "mean_gap_post_ci95": percentile_ci(np.array(gaps_c)),
            },
            "fine_bands": {
                **fine,
                "mean_gap_post_ci95": percentile_ci(np.array(gaps_f)),
            },
            "note": (
                "each p_max band mapped to its pooled accuracy, fitted in-sample "
                "on both periods; see the recalibration analysis for a bound "
                "over every function of p_max"
            ),
        },
        "clusters": {"pre_months": len(pre_months), "post_months": len(post_months)},
        "bootstrap": f"months resampled within each period; cutoff fixed at {CUTOFF_MONTH}",
    }


# --- 2. prevalence sensitivity --------------------------------------------------


def paired_rows(e: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pair each question's plain answer with its answer given a "not known" option.

    Args:
        e: Cutoff rows of the follow-up questions experiment, with ``unit``
            (``cutoff:<cond>:<question>``), ``cond`` (``base`` and
            ``unknown`` are used), ``gold``, ``top``, ``dist``, ``month``,
            ``post``, ``p_max`` and ``p_known``.

    Returns:
        Arrays aligned on the plain rows' order: ``month``, ``post``,
        ``base_ok`` (plain answer right), ``p_max``, ``p_known``,
        ``answered`` (the option was not chosen) and ``opt_ok`` (the
        option-condition answer forced to yes/no is right).
    """
    e = e.copy()
    e["key"] = e["unit"].str.replace(r"^cutoff:[a-z_]+:", "", regex=True)
    base = e[e["cond"] == "base"].set_index("key")
    unk = e[e["cond"] == "unknown"].set_index("key").loc[base.index]
    gold = base["gold"].to_numpy()
    forced = np.array(["yes" if d["yes"] >= d["no"] else "no" for d in unk["dist"]])
    return {
        "month": base["month"].astype(str).to_numpy(),
        "post": base["post"].to_numpy(bool),
        "base_ok": base["top"].to_numpy() == gold,
        "p_max": base["p_max"].to_numpy(float),
        "p_known": base["p_known"].to_numpy(float),
        "answered": unk["top"].to_numpy() != "unknown",
        "opt_ok": forced == gold,
    }


def at_share(
    x: dict[str, np.ndarray], share: float, idx: np.ndarray | None = None
) -> dict[str, float]:
    """Option vs thresholds when a ``share`` of traffic is post-cutoff.

    Args:
        x: Output of :func:`paired_rows`.
        share: Post-cutoff share of questions.
        idx: Item indices to use (for the bootstrap); all if ``None``.

    Returns:
        Coverage and accuracy of the option and of both thresholds at that
        coverage, with the thresholds' achieved coverage.
    """
    i = np.arange(len(x["post"])) if idx is None else idx
    post = x["post"][i]
    n_post, n_pre = post.sum(), (~post).sum()
    w = np.where(post, share / max(n_post, 1), (1 - share) / max(n_pre, 1))
    answered = x["answered"][i]
    cov = float((w * answered).sum() / w.sum())
    acc_opt = float(
        (w * answered * x["opt_ok"][i]).sum() / max((w * answered).sum(), 1e-12)
    )
    ok_b = x["base_ok"][i]
    acc_pm, cov_pm = weighted_accuracy_at(x["p_max"][i], ok_b, w, cov)
    acc_pk, cov_pk = weighted_accuracy_at(x["p_known"][i], ok_b, w, cov)
    return {
        "coverage": cov,
        "option": acc_opt,
        "p_max": acc_pm,
        "p_known": acc_pk,
        "achieved_coverage_p_max": cov_pm,
        "achieved_coverage_p_known": cov_pk,
    }


def prevalence(e: pd.DataFrame, b: int, rng: np.random.Generator) -> dict[str, Any]:
    """The matched-coverage comparison across post-cutoff shares.

    Args:
        e: Cutoff rows (see :func:`paired_rows`).
        b: Bootstrap resamples at each reported share.
        rng: Random generator (one moving-block resample per draw, shares in
            :data:`REPORT_SHARES` order).

    Returns:
        A curve over shares; at selected shares the option minus each
        threshold with 95 % intervals; the shares where the option is more
        accurate than each threshold.
    """
    x = paired_rows(e)
    curve = {f"{s:.2f}": at_share(x, s) for s in SHARES}
    report = {}
    for s in REPORT_SHARES:
        pt = at_share(x, s)
        d_pm, d_pk = [], []
        for _ in range(b):
            idx = month_block_indices(x["month"], x["post"], rng, BLOCK_MONTHS)
            r = at_share(x, s, idx)
            d_pm.append(r["option"] - r["p_max"])
            d_pk.append(r["option"] - r["p_known"])
        report[f"{s:.3f}"] = {
            **pt,
            "option_minus_p_max": pt["option"] - pt["p_max"],
            "option_minus_p_max_ci95": percentile_ci(np.array(d_pm)),
            "option_minus_p_known": pt["option"] - pt["p_known"],
            "option_minus_p_known_ci95": percentile_ci(np.array(d_pk)),
        }

    def wins(key: str) -> list[float]:
        return [float(k) for k in curve if curve[k]["option"] > curve[k][key]]

    return {
        "sample": {"post": int(x["post"].sum()), "pre": int((~x["post"]).sum())},
        "panel_share": PANEL_SHARE,
        "curve": curve,
        "at_selected_shares": report,
        "shares_where_option_beats_p_max": wins("p_max"),
        "shares_where_option_beats_p_known": wins("p_known"),
        "note": (
            f"period-stratified moving-block bootstrap over months ({BLOCK_MONTHS}-month "
            "blocks), conditional on the observed three-call answers; coverage "
            "matched exactly"
        ),
    }


# --- 3. figure ---------------------------------------------------------------


def figure(eq: dict[str, Any], prev: dict[str, Any], out: Path) -> Path:
    """Draw accuracy at equal confidence and the prevalence curve.

    Args:
        eq: Output of :func:`equal_confidence`.
        prev: Output of :func:`prevalence`.
        out: PNG path.

    Returns:
        ``out``.
    """
    from beyond_answer_confidence.reporting.figures import (
        IDEAL,
        S1,
        S2,
        S3,
        SURFACE,
        TEXT_2,
        _line,
        _plt,
        _save,
        _style,
    )

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURFACE)
    ax = axes[0]
    bands = eq["bands"]
    xs = np.arange(len(bands))
    for key, color, label in (
        ("pre", S1, "before the cutoff"),
        ("post", S2, "after the cutoff"),
    ):
        ys = np.array([bd[f"accuracy_{key}"] for bd in bands], float)
        lo_hi = [bd[f"accuracy_{key}_ci95"] for bd in bands]
        ax.errorbar(
            xs,
            ys,
            yerr=[ys - [a for a, _ in lo_hi], [c for _, c in lo_hi] - ys],
            fmt="none",
            ecolor=color,
            capsize=3,
        )
        _line(ax, xs, ys, color, label)
    mids = [0.55, 0.65, 0.75, 0.85, 0.95]
    ax.plot(xs, mids, color=IDEAL, lw=1.2, ls="--", label="accuracy = confidence")
    ax.set_xticks(xs, BAND_LABELS, fontsize=8.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("answer probability (band)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("accuracy (95 % month-cluster interval)", color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "a. Same confidence, different accuracy")
    ax = axes[1]
    c = prev["curve"]
    sx = [float(k) for k in c]
    _line(ax, sx, [c[k]["option"] for k in c], S3, '"not known" option', marker="s")
    _line(ax, sx, [c[k]["p_max"] for k in c], S1, "confidence threshold")
    _line(
        ax, sx, [c[k]["p_known"] for k in c], S2, '"do you know?" threshold', marker="^"
    )
    ax.axvline(prev["panel_share"], color=IDEAL, lw=1, ls=":")
    ax.axvline(0.5, color=IDEAL, lw=1, ls="--")
    ax.text(
        prev["panel_share"] - 0.01,
        0.92,
        "unreweighted dated\npanel share",
        fontsize=7.5,
        color=TEXT_2,
        va="top",
        ha="right",
    )
    ax.text(0.51, 0.92, "balanced\nsample", fontsize=7.5, color=TEXT_2, va="top")
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("share of questions from after the cutoff", color=TEXT_2, fontsize=9)
    ax.set_ylabel(
        "accuracy on answered questions (matched coverage)", color=TEXT_2, fontsize=9
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "b. Abstention benefit depends on the mix")
    return _save(fig, out)


# --- runner -------------------------------------------------------------------


def news_rows(settings: Settings, *, precise_float: bool = False) -> pd.DataFrame:
    """Dated yes/no news rows of the knowledge-boundary experiment, with ``post``.

    Args:
        settings: Run settings (locate the outputs).
        precise_float: Parse floats round-trip exactly (see
            :mod:`beyond_answer_confidence.analyses.rows`).

    Returns:
        Rows of set ``oracle_tf`` with ``post = month >= CUTOFF_MONTH``.
    """
    rows = experiment_frame(
        settings,
        "knowledge_boundary",
        dtype={"month": str},
        precise_float=precise_float,
    )
    tf = rows[rows["set"] == NEWS_SET].copy()
    tf["post"] = tf["month"] >= CUTOFF_MONTH
    return tf


def cutoff_rows(settings: Settings, *, precise_float: bool = False) -> pd.DataFrame:
    """Cutoff rows (units ``cutoff:*``) of the follow-up questions experiment.

    Args:
        settings: Run settings (locate the outputs).
        precise_float: Parse floats round-trip exactly.

    Returns:
        The rows.
    """
    rows = experiment_frame(
        settings,
        "follow_up_questions",
        dtype={"month": str},
        precise_float=precise_float,
    )
    return rows[rows["unit"].astype(str).str.startswith("cutoff:")].copy()


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run both analyses and write the summary (and figure).

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        ``equal_confidence`` and ``prevalence`` results.
    """
    rng = np.random.default_rng(config.seed)
    res = {
        "equal_confidence": equal_confidence(
            news_rows(settings, precise_float=config.precise_float),
            config.bootstrap,
            rng,
        ),
        "prevalence": prevalence(
            cutoff_rows(settings, precise_float=config.precise_float),
            config.bootstrap,
            rng,
        ),
    }
    out = settings.out(NAME)
    write_json(out / "summary.json", {"config": as_dict(config), **res})
    if config.figures:
        figure(
            res["equal_confidence"], res["prevalence"], out / "figures" / f"{NAME}.png"
        )
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Accuracy at equal confidence across the cutoff; abstention vs mix",
    config_type=Config,
    run=run,
)
