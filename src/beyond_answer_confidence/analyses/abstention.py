"""Abstention around the knowledge cutoff, and renormalised odds readings.

Offline; reads saved rows. One generator for sections 1-3, in order:

1. **The "not known" option at matched coverage.** Offering a "not known"
   option removes most confident post-cutoff errors, but the model then
   answers few of those questions, and refusing everything would also
   remove every error. The option is therefore compared with two
   thresholds at the *same* answer rate on the same items without the
   option: answering the most confident questions by the answer
   probability, or by P(known). Paired item-bootstrap intervals for the
   accuracy difference and risk-coverage areas (AURC) for all three scores,
   for all questions and for each side of the cutoff.
2. **Separability of answer confidence.** AUROC of the answer probability
   and of 1 - P(known) for post- vs pre-cutoff questions, with a
   month-cluster bootstrap. An increasing map of the answer probability
   keeps its ranking, so it cannot move post-cutoff confidence below
   pre-cutoff confidence if the raw ranking puts post-cutoff questions
   higher.
3. **P(known) over time with flexible trends.** The step in monthly P(known)
   at the cutoff under linear, quadratic and natural-spline trends, with
   Newey-West (HAC) and moving-block residual-bootstrap intervals, and
   AIC/BIC with and without the step. Descriptive.
4. **Renormalised per-outcome readings.** Per-outcome yes/no and score
   readings of stated odds do not sum to 1; renormalising them is scored
   against the stated odds (mean absolute error per option, total
   variation, KL divergence).

Inputs: ``follow_up_questions`` rows (units ``cutoff:*`` and ``odds:*``) and
the dated yes/no rows of ``knowledge_boundary``. Output:
``<output_dir>/abstention/summary.json``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.analyses.cutoff_calibration import CUTOFF_MONTH, NEWS_SET
from beyond_answer_confidence.analyses.rows import experiment_frame
from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.experiments.base import Analysis, read_jsonl, write_json
from beyond_answer_confidence.metrics.ranking import aurc, auroc, selective_at
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.resampling import cluster_bootstrap, percentile_ci
from beyond_answer_confidence.tasks.follow_up import percent_reading
from beyond_answer_confidence.tasks.synthetic import make_scenario

NAME = "abstention"
TREND_SPECS = ("linear", "quadratic", "spline_4", "spline_6")
"""Trend models for monthly P(known) (``spline_<df>``: natural cubic spline)."""


@dataclass(frozen=True)
class Config:
    """Options.

    Attributes:
        bootstrap: Bootstrap resamples per interval.
        seed: Seed.
        confident: Answer probability at or above which an error counts as
            confident.
        block_months: Block length of the residual bootstrap over months.
    """

    bootstrap: int = 2000
    seed: int = 0
    confident: float = 0.9
    block_months: int = 6


# --- 1. abstention ---------------------------------------------------------------


def abstention(
    e: pd.DataFrame, b: int, rng: np.random.Generator, confident: float = 0.9
) -> dict[str, Any]:
    """Compare the "not known" option with thresholds at matched coverage.

    Args:
        e: Cutoff rows of the follow-up questions experiment (``unit``
            ``cutoff:<cond>:<question>``; conditions ``base`` and
            ``unknown`` are used).
        b: Bootstrap resamples per subset.
        rng: Random generator.
        confident: Confidence threshold for counting confident errors.

    Returns:
        Per subset (``all``, ``post``, ``pre``): the option's coverage and
        accuracy, both thresholds at the same coverage, confident-error
        counts, AURC per score and paired bootstrap intervals of the
        accuracy difference (option minus threshold).
    """
    e = e.copy()
    e["key"] = e["unit"].str.replace(r"^cutoff:[a-z_]+:", "", regex=True)
    base = e[e["cond"] == "base"].set_index("key")
    unk = e[e["cond"] == "unknown"].set_index("key").loc[base.index]
    gold = base["gold"].to_numpy()
    base_ok = base["top"].to_numpy() == gold
    p_unknown = np.array([d["unknown"] for d in unk["dist"]])
    forced = np.array(["yes" if d["yes"] >= d["no"] else "no" for d in unk["dist"]])
    unk_ok = forced == gold
    answered = unk["top"].to_numpy() != "unknown"
    post = base["post"].to_numpy(bool)
    out: dict[str, Any] = {}
    for name, mask in (
        ("all", np.ones(len(base), bool)),
        ("post", post),
        ("pre", ~post),
    ):
        cov = float(answered[mask].mean())
        pm = base["p_max"].to_numpy(float)[mask]
        pk = base["p_known"].to_numpy(float)[mask]
        ok_b = base_ok[mask]
        res: dict[str, Any] = {
            "items": int(mask.sum()),
            "not_known_option": {
                "answered": int(answered[mask].sum()),
                "coverage": cov,
                "selective_accuracy": float(unk_ok[mask][answered[mask]].mean())
                if answered[mask].any()
                else None,
                "errors": int((answered[mask] & ~unk_ok[mask]).sum()),
                "confident_errors": int(
                    (
                        answered[mask]
                        & ~unk_ok[mask]
                        & (unk["p_max"].to_numpy()[mask] >= confident)
                    ).sum()
                ),
            },
            "threshold_p_max": selective_at(pm, ok_b, cov, pm, confident),
            "threshold_p_known": selective_at(pk, ok_b, cov, pm, confident),
            "forced_answers": {
                "errors": int((~ok_b).sum()),
                "confident_errors": int(((~ok_b) & (pm >= confident)).sum()),
            },
            "aurc": {
                "p_max_without_option": aurc(pm, ok_b),
                "p_known_without_option": aurc(pk, ok_b),
                "1-P(not known)_with_option": aurc(1 - p_unknown[mask], unk_ok[mask]),
            },
        }
        idx = np.flatnonzero(mask)
        diffs: dict[str, list[float]] = {"p_max": [], "p_known": []}
        for _ in range(b):
            s = rng.choice(idx, len(idx), replace=True)
            if not answered[s].any():
                continue
            c = float(answered[s].mean())
            acc_opt = float(unk_ok[s][answered[s]].mean())
            for key in ("p_max", "p_known"):
                r = selective_at(base[key].to_numpy(float)[s], base_ok[s], c)
                if r["selective_accuracy"] is not None:
                    diffs[key].append(acc_opt - r["selective_accuracy"])
        res["accuracy_option_minus_baseline"] = {
            k: {"mean": float(np.mean(v)), "ci95": percentile_ci(np.array(v))}
            for k, v in diffs.items()
            if v
        }
        out[name] = res
    return out


# --- 2. separability ---------------------------------------------------------------


def _auroc_of(score: np.ndarray, positive: np.ndarray) -> Callable[[np.ndarray], float]:
    def stat(i: np.ndarray) -> float:
        j = i.astype(int)
        return auroc(score[j], positive[j])

    return stat


def separability(tf: pd.DataFrame, b: int, rng: np.random.Generator) -> dict[str, Any]:
    """AUROC of answer confidence and 1 - P(known) for post- vs pre-cutoff questions.

    Args:
        tf: Dated yes/no rows with ``month``, ``p_max`` and ``p_known``.
        b: Bootstrap resamples (month clusters).
        rng: Random generator.

    Returns:
        Per score: AUROC with its month-cluster interval and the mean score
        per period.
    """
    post = (tf["month"] >= CUTOFF_MONTH).to_numpy()
    months = tf["month"].to_numpy()
    idx = np.arange(len(tf))
    out: dict[str, Any] = {}
    for name, score in (
        ("p_max", tf["p_max"].to_numpy(float)),
        ("1-p_known", 1 - tf["p_known"].to_numpy(float)),
    ):
        boot = cluster_bootstrap(idx, months, b, rng, stat=_auroc_of(score, post))
        out[name] = {
            "auroc_post_vs_pre": auroc(score, post),
            "ci95": percentile_ci(boot),
            "mean_pre": float(score[~post].mean()),
            "mean_post": float(score[post].mean()),
        }
    return out


# --- 3. P(known) trend ---------------------------------------------------------------


def trend_design(t: np.ndarray, spec: str, step: np.ndarray | None) -> np.ndarray:
    """Design matrix of a trend model, optionally with a step column last.

    Args:
        t: Time per month (years).
        spec: One of :data:`TREND_SPECS`.
        step: Step indicator per month, or ``None``.

    Returns:
        Columns: intercept, trend terms, then the step (if given).
    """
    if spec == "linear":
        cols = [np.ones_like(t), t]
    elif spec == "quadratic":
        cols = [np.ones_like(t), t, t**2]
    else:
        import patsy

        df = int(spec.split("_")[1])
        basis = np.asarray(patsy.dmatrix(f"cr(t, df={df}) - 1", {"t": t}))
        cols = [np.ones_like(t), *basis.T[:-1]]  # one column dropped for the intercept
    if step is not None:
        cols.append(step)
    return np.column_stack(cols)


def known_trend(
    tf: pd.DataFrame, b: int, rng: np.random.Generator, block: int = 6
) -> dict[str, Any]:
    """The step in monthly P(known) at the cutoff under several trend models.

    Args:
        tf: Dated yes/no rows with ``month`` and ``p_known``.
        b: Block-bootstrap resamples per model.
        rng: Random generator.
        block: Block length in months.

    Returns:
        Per trend model: the step, HAC and block-bootstrap intervals, and
        AIC/BIC with and without the step.
    """
    import statsmodels.api as sm

    monthly = (
        tf.groupby("month")
        .agg(y=("p_known", "mean"), n=("p_known", "size"))
        .sort_index()
    )
    t = np.array([(int(m[:4]) - 2020) + (int(m[5:7]) - 1) / 12 for m in monthly.index])
    step = (monthly.index >= CUTOFF_MONTH).astype(float)
    y, w = monthly["y"].to_numpy(float), monthly["n"].to_numpy(float)
    out: dict[str, Any] = {"months": len(y), "block_months": block}
    for spec in TREND_SPECS:
        x = trend_design(t, spec, step)
        fit = sm.WLS(y, x, weights=w).fit()
        hac = sm.WLS(y, x, weights=w).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        fit0 = sm.WLS(y, trend_design(t, spec, None), weights=w).fit()
        resid, fitted = fit.resid, fit.fittedvalues
        boots = []
        starts = np.arange(len(y) - block + 1)
        for _ in range(b):
            pieces = [
                resid[s : s + block] for s in rng.choice(starts, -(-len(y) // block))
            ]
            ystar = fitted + np.concatenate(pieces)[: len(y)]
            boots.append(sm.WLS(ystar, x, weights=w).fit().params[-1])
        ci = hac.conf_int()[-1]
        out[spec] = {
            "step": float(fit.params[-1]),
            "step_ci95_hac": [float(ci[0]), float(ci[1])],
            "step_ci95_block_bootstrap": percentile_ci(np.array(boots)),
            "aic_with_step": float(fit.aic),
            "aic_without_step": float(fit0.aic),
            "bic_with_step": float(fit.bic),
            "bic_without_step": float(fit0.bic),
        }
    return out


# --- 4. renormalised readings -------------------------------------------------------


READING_COLUMNS = (
    "sum",
    "mae_raw",
    "mae_renormalised",
    "tv_renormalised",
    "kl_renormalised",
)


def renormalised(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-outcome readings, raw and renormalised, against the stated odds.

    Args:
        rows: Stated-odds rows (``scenario``, ``format`` and the readings:
            ``p_opt<k>`` for ``noul``, ``s_opt<k>`` for ``score``, ``dist``
            for ``instructed``).

    Returns:
        Per format: mean reading sum, mean absolute error per option (raw
        and renormalised), total variation and KL(stated || renormalised),
        overall and by scenario profile.
    """
    per: dict[str, list[dict[str, Any]]] = {"noul": [], "score": [], "instructed": []}
    for r in rows:
        sc = make_scenario(r["scenario"])
        star = np.array([sc.p_star[o] for o in sc.options])
        if r["format"] == "noul":
            read = np.array([r[f"p_opt{k}"] for k in range(4)])
        elif r["format"] == "score":
            read = np.array([percent_reading(r[f"s_opt{k}"]) for k in range(4)])
        else:
            read = np.array([r["dist"][o] for o in sc.options])
        norm = read / max(read.sum(), 1e-9)
        nz = star > 0
        per[r["format"]].append(
            {
                "profile": sc.profile,
                "sum": float(read.sum()),
                "mae_raw": float(np.abs(read - star).mean()),
                "mae_renormalised": float(np.abs(norm - star).mean()),
                "tv_renormalised": float(0.5 * np.abs(norm - star).sum()),
                "kl_renormalised": float(
                    (star[nz] * np.log(star[nz] / np.maximum(norm[nz], 1e-6))).sum()
                ),
            }
        )
    out: dict[str, Any] = {}
    for fmt, xs in per.items():
        df = pd.DataFrame(xs)
        cols = list(READING_COLUMNS)
        out[fmt] = {
            "n": len(df),
            **{c: float(df[c].mean()) for c in cols},
            "by_profile": {
                p: {c: float(g[c].mean()) for c in cols}
                for p, g in df.groupby("profile")
            },
        }
    return out


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Run the four sections and write the summary.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        ``abstention``, ``separability``, ``known_trend`` and ``renormalised``.
    """
    rng = np.random.default_rng(config.seed)
    follow = experiment_frame(settings, "follow_up_questions", dtype={"month": str})
    e = follow[follow["unit"].astype(str).str.startswith("cutoff:")]
    boundary = experiment_frame(settings, "knowledge_boundary", dtype={"month": str})
    tf = boundary[boundary["set"] == NEWS_SET].reset_index(drop=True)
    odds = [
        r
        for r in read_jsonl(settings.out("follow_up_questions") / "rows.jsonl")
        if str(r.get("unit", "")).startswith("odds:")
    ]
    res = {
        "abstention": abstention(e, config.bootstrap, rng, config.confident),
        "separability": separability(tf, config.bootstrap, rng),
        "known_trend": known_trend(tf, config.bootstrap, rng, config.block_months),
        "renormalised": renormalised(odds),
    }
    write_json(settings.out(NAME) / "summary.json", {"config": as_dict(config), **res})
    return res


ANALYSIS = Analysis(
    name=NAME,
    summary="Abstention at matched coverage, separability, P(known) trend, readings",
    config_type=Config,
    run=run,
)
