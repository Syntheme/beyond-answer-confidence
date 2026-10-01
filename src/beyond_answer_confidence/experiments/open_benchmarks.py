"""Open benchmark variants: abstention, verification and ambiguity.

Three adaptations of public benchmarks:

- SimpleQA Verified with a fifth "I don't know the answer" option: does its
  probability flag the questions answered wrongly without it?
- TriviaQA as verification ("Is the proposed answer correct?"), with the
  gold answer and one distractor shown in turn;
- AmbigQA: the options are the distinct answers of a question's
  disambiguated readings plus distractors; does probability spread over the
  valid answers? Unambiguous questions are the control.

The SimpleQA analysis reads the benchmarks experiment's rows (the same
questions without the extra option).
"""

import dataclasses
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyond_answer_confidence.config import as_dict
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.distractors import (
    DistractorPicker,
    distractor_picker,
    normalise_answer,
)
from beyond_answer_confidence.experiments.base import (
    Experiment,
    backend_for,
    read_jsonl,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.runner import run_items
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.stats.inference import (
    auroc_vs_threshold,
    holm_adjust_tails,
    mean_interval,
    mean_interval_lower,
)
from beyond_answer_confidence.stats.resampling import smece_interval
from beyond_answer_confidence.tasks.benchmarks import (
    ambig_label,
    simpleqa_items,
    triviaqa_items,
)
from beyond_answer_confidence.tasks.multiple_choice import LETTERS
from beyond_answer_confidence.tasks.schema import Item, Task, item_tasks

logger = logging.getLogger(__name__)

NAME = "open_benchmarks"

BENCHMARKS_EXPERIMENT = "benchmarks"
"""Experiment whose SimpleQA rows give correctness without the extra option."""

IDK = "I don't know the answer"
"""Description of the abstention option."""

VERIFY_QUESTION = "Is the proposed answer correct?"
"""Yes/no verification question."""

AMBIG_INSTRUCTIONS = "Choose the correct answer to the question."


@dataclass(frozen=True)
class Config:
    """Options of the open-benchmark variants.

    Attributes:
        replicates: Replicates per item.
        seed: Seed of the source item sets and the AmbigQA option order.
        verification_questions: TriviaQA questions verified.
        verification_seed: Seed of the TriviaQA sample.
        bootstrap: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        idk_auroc_reference: Reference AUROC of P(I don't know) for errors.
        confident_level: P(correct) counted as confident acceptance.
        accept_reference: Reference share of false answers accepted
            confidently.
        smece_reference: Reference SmoothECE of verification.
        spread_reference: Reference normalised entropy over valid answers.
    """

    replicates: int = 3
    seed: int = 0
    verification_questions: int = 2000
    verification_seed: int = 0
    bootstrap: int = 2000
    bootstrap_seed: int = 0
    idk_auroc_reference: float = 0.75
    confident_level: float = 0.9
    accept_reference: float = 0.05
    smece_reference: float = 0.05
    spread_reference: float = 0.5


def simpleqa_idk_items(simpleqa: Sequence[Item]) -> list[Item]:
    """SimpleQA items with an extra "I don't know" option.

    Args:
        simpleqa: SimpleQA multiple-choice items.

    Returns:
        Items (units ``idk:<unit>``; info ``idk`` = the abstention letter,
        ``benchmark_unit`` = the original unit).

    Raises:
        ValueError: If an item has no option descriptions.
    """
    out = []
    for it in simpleqa:
        if it.descriptions is None:
            raise ValueError(f"{it.unit} has no option descriptions")
        letter = LETTERS[len(it.options)]
        out.append(
            dataclasses.replace(
                it,
                unit=f"idk:{it.unit}",
                options=(*it.options, letter),
                descriptions={**it.descriptions, letter: IDK},
                info={
                    **it.info,
                    "set": "simpleqa_idk",
                    "idk": letter,
                    "benchmark_unit": it.unit,
                },
            )
        )
    return out


def verification_items(
    trivia: Sequence[Item], n: int = 2000, seed: int = 0
) -> list[Item]:
    """TriviaQA verification items: the gold and the first other option, in turn.

    Args:
        trivia: TriviaQA multiple-choice items.
        n: Questions sampled (``random.Random(seed).sample``).
        seed: Sampling seed.

    Returns:
        Two yes/no items per question (units ``verify:<unit>:true|false``).

    Raises:
        ValueError: If an item has no descriptions or no gold.
    """
    out = []
    for it in random.Random(seed).sample(list(trivia), n):  # noqa: S311  # nosec B311
        if it.descriptions is None or it.gold is None:
            raise ValueError(f"{it.unit} needs descriptions and a gold option")
        wrong = next(o for o in it.options if o != it.gold)
        for truth, letter in ((True, it.gold), (False, wrong)):
            out.append(
                Item(
                    unit=f"verify:{it.unit}:{'true' if truth else 'false'}",
                    state={
                        "question": it.state["question"],
                        "proposed_answer": it.descriptions[letter],
                    },
                    nouls={"correct": VERIFY_QUESTION},
                    info={
                        "set": "trivia_verify",
                        "truth": truth,
                        "question_unit": it.unit,
                    },
                )
            )
    return out


def _first_answer(raw: Any) -> str | None:
    vals = [str(a).strip() for a in raw if str(a).strip()]
    return vals[0] if vals else None


def ambiguity_candidates(ambigqa: pd.DataFrame) -> list[dict[str, Any]]:
    """AmbigQA questions usable as spread items or controls.

    Ambiguous questions keep the first answer of each disambiguated reading
    when there are two to four distinct answers (distinct also after
    normalisation); unambiguous ones keep their first answer.

    Args:
        ambigqa: Table as returned by
            :func:`beyond_answer_confidence.data.loaders.load_ambigqa_validation`.

    Returns:
        Dicts with ``id``, ``question``, ``valid`` answers and ``kind``
        (``ambiguous`` or ``single``).
    """
    rows = []
    for r in loaders.table_rows(ambigqa):
        label = ambig_label(list(r.annotations["type"]))
        if label == "ambiguous":
            qp = r.annotations["qaPairs"][0]
            answers = [_first_answer(a) for a in qp["answer"]]
            valid = list(dict.fromkeys(a for a in answers if a))
            norm = list(dict.fromkeys(normalise_answer(a) for a in valid))
            if 2 <= len(norm) <= 4 and len(norm) == len(valid):
                rows.append(
                    {
                        "id": r.id,
                        "question": r.question,
                        "valid": valid,
                        "kind": "ambiguous",
                    }
                )
        elif label == "single":
            gold = _first_answer(r.annotations["answer"][0])
            if gold:
                rows.append(
                    {
                        "id": r.id,
                        "question": r.question,
                        "valid": [gold],
                        "kind": "single",
                    }
                )
    return rows


def ambiguity_items(
    candidates: Sequence[Mapping[str, Any]], pick: DistractorPicker, seed: int = 0
) -> list[Item]:
    """Spread items (ambiguous) and controls (unambiguous).

    Ambiguous questions get two distractors, controls three; the options
    are shuffled per question.

    Args:
        candidates: Output of :func:`ambiguity_candidates`.
        pick: Distractor picker (nearest answers from the pool of all valid
            answers).
        seed: Option-order seed.

    Returns:
        Items (units ``ambig:<id>``; info ``valid`` = valid letters).
    """
    pool = sorted({a for row in candidates for a in row["valid"]})
    golds = [row["valid"][0] for row in candidates]
    excludes = [row["valid"] for row in candidates]
    near = pick(golds, excludes, pool)
    items = []
    for row, dis in zip(candidates, near, strict=True):
        if dis is None:
            continue
        n_dis = 2 if row["kind"] == "ambiguous" else 3
        answers = [*row["valid"], *dis[:n_dis]]
        rng = random.Random(f"ambig:{seed}:{row['id']}")  # noqa: S311  # nosec B311
        order = list(range(len(answers)))
        rng.shuffle(order)
        letters = LETTERS[: len(answers)]
        desc = {letters[pos]: answers[i] for pos, i in enumerate(order)}
        valid_letters = [letters[order.index(i)] for i in range(len(row["valid"]))]
        items.append(
            Item(
                unit=f"ambig:{row['id']}",
                state={"question": row["question"]},
                options=letters,
                gold=valid_letters[0],
                instructions=AMBIG_INSTRUCTIONS,
                descriptions=desc,
                info={
                    "set": "ambig_spread",
                    "kind": row["kind"],
                    "valid": valid_letters,
                },
            )
        )
    return items


def build_items(settings: Settings, config: Config) -> list[Item]:
    """Load the datasets and build every item.

    Args:
        settings: Run settings (data directory).
        config: Options.

    Returns:
        SimpleQA abstention items, TriviaQA verification items, AmbigQA items.
    """
    d = settings.data_dir
    pick = distractor_picker(d)
    trivia = triviaqa_items(loaders.load_triviaqa_validation(d), pick, config.seed)
    return (
        simpleqa_idk_items(simpleqa_items(loaders.load_simpleqa(d), pick, config.seed))
        + verification_items(
            trivia, config.verification_questions, config.verification_seed
        )
        + ambiguity_items(
            ambiguity_candidates(loaders.load_ambigqa_validation(d)), pick, config.seed
        )
    )


def requests(settings: Settings, config: Config) -> list[Task]:
    """Every task of the experiment.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The tasks.
    """
    return item_tasks(build_items(settings, config), config.replicates)


def valid_spread(dist: Mapping[str, float], valid: Sequence[str]) -> float:
    """Normalised entropy of the probability on the valid answers.

    Args:
        dist: Choice distribution.
        valid: Valid option keys (at least two).

    Returns:
        Entropy of the renormalised valid-answer probabilities divided by
        ``log(len(valid))``; 0 if they carry no probability.
    """
    v = np.array([dist[k] for k in valid])
    if v.sum() <= 0:
        return 0.0
    v = v / v.sum()
    nz = v[v > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(v)))


def analyse(
    rows: list[dict[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
    config: Config,
) -> dict[str, Any]:
    """Abstention AUROC, verification acceptance and calibration, answer spread.

    Bootstrap draws (one generator, in this order): AUROC of P(I don't know)
    for questions answered wrongly without the option; share of false
    answers accepted with P(correct) at or above the confident level;
    SmoothECE of verification; mean spread over valid answers.

    Args:
        rows: Scored rows.
        benchmark_rows: Benchmarks-experiment rows (``unit``, ``correct``).
        config: Options.

    Returns:
        Effect sizes with intervals (Holm-adjusted tail shares) and
        descriptive summaries per variant.
    """
    b = config.bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    df = pd.DataFrame(rows)
    old = {r["unit"]: r for r in benchmark_rows}
    sq = df[df["set"] == "simpleqa_idk"].copy()
    sq["p_idk"] = [d[k] for d, k in zip(sq["dist"], sq["idk"], strict=True)]
    sq["benchmark_wrong"] = [not old[u]["correct"] for u in sq["benchmark_unit"]]
    idk = auroc_vs_threshold(
        sq["p_idk"].to_numpy(float),
        sq["benchmark_wrong"].to_numpy(bool),
        config.idk_auroc_reference,
        b,
        rng,
    )
    ver = df[df["set"] == "trivia_verify"].copy()
    p = ver["p_correct"].to_numpy(float)
    truth = ver["truth"].to_numpy(bool)
    accepted = mean_interval(
        (p[~truth] >= config.confident_level).astype(float),
        b,
        rng,
        threshold=config.accept_reference,
    )
    conf = np.maximum(p, 1 - p)
    right = (p >= 0.5) == truth
    calibration = smece_interval(conf, right, b, rng, threshold=config.smece_reference)
    am = df[df["set"] == "ambig_spread"]
    amb = am[am["kind"] == "ambiguous"]
    spreads = np.array(
        [valid_spread(d, v) for d, v in zip(amb["dist"], amb["valid"], strict=True)]
    )
    spread = mean_interval_lower(spreads, b, rng, config.spread_reference)
    res: dict[str, Any] = holm_adjust_tails(
        {
            "idk_flags_benchmark_errors": idk,
            "false_answers_confidently_accepted": accepted,
            "verification_calibration": calibration,
            "ambiguity_spreads_probability": spread,
        }
    )
    abstained = np.array([t == k for t, k in zip(sq["top"], sq["idk"], strict=True)])
    answered = sq[~abstained]
    single = am[am["kind"] == "single"]
    res |= {
        "simpleqa_idk": {
            "abstention_rate": float(np.mean(abstained)),
            "mean_p_idk": float(sq["p_idk"].mean()),
            "accuracy_answered": float(answered["correct"].mean())
            if len(answered)
            else None,
            "coverage": len(answered) / len(sq),
            "multiple_choice_accuracy": float(
                np.mean([old[u]["correct"] for u in sq["benchmark_unit"]])
            ),
        },
        "trivia_verify": {
            "auroc_true_vs_false": auroc(p, truth),
            "mean_p_true_answers": float(p[truth].mean()),
            "mean_p_false_answers": float(p[~truth].mean()),
            "share_true_rejected_le_0.1": float(np.mean(p[truth] <= 0.1)),
        },
        "ambig_spread": {
            "n_ambiguous": len(amb),
            "mean_mass_on_valid": float(
                np.mean(
                    [
                        sum(d[k] for k in v)
                        for d, v in zip(amb["dist"], amb["valid"], strict=True)
                    ]
                )
            ),
            "mean_p_max": float(amb["p_max"].mean()),
            "mean_n_valid": float(np.mean([len(v) for v in amb["valid"]])),
            "control_accuracy": float(single["correct"].mean()),
            "control_mean_p_max": float(single["p_max"].mean()),
        },
    }
    return res


def run(settings: Settings, config: Config) -> dict[str, Any]:
    """Collect (cache first), score, analyse and write the outputs.

    Args:
        settings: Run settings.
        config: Options.

    Returns:
        The summary (``results`` only when every item is complete and the
        benchmarks rows exist).
    """
    rows, book = run_items(
        build_items(settings, config),
        cache=settings.cache_file(NAME),
        model=settings.model,
        replicates=config.replicates,
        backend=backend_for(settings),
        max_input_tokens=settings.max_input_tokens,
        concurrency=settings.concurrency,
        label=NAME,
    )
    out = settings.out(NAME)
    write_jsonl(out / "rows.jsonl", rows)
    summary: dict[str, Any] = {"config": as_dict(config), **book}
    reference: Path = settings.out(BENCHMARKS_EXPERIMENT) / "rows.jsonl"
    if book["complete_items"] != book["items"]:
        logger.warning("%s: incomplete items; no analysis", NAME)
    elif not reference.exists():
        logger.warning("%s: %s not found; run benchmarks first", NAME, reference)
    else:
        summary["results"] = analyse(rows, read_jsonl(reference), config)
    write_json(out / "summary.json", summary)
    return summary


EXPERIMENT = Experiment(
    name=NAME,
    summary="SimpleQA with an abstain option, TriviaQA verification, AmbigQA spread",
    config_type=Config,
    requests=requests,
    run=run,
)
