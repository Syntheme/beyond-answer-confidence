import random
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.backends.base import RawAnswer, Request
from beyond_answer_confidence.data import distractors, loaders
from beyond_answer_confidence.experiments import fresh_names as fx
from beyond_answer_confidence.experiments import paraphrase_stability as px
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks import fresh_names as fn
from beyond_answer_confidence.tasks import paraphrases as pp
from beyond_answer_confidence.tasks.multiple_choice import KNOWN_QUESTION, LETTERS
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.shortcuts import (
    PERSON_RELATION_ORDER,
    is_person_name,
)

# --- synthetic PopQA tables ------------------------------------------------------


def _word(k: int) -> str:
    """Lower-case letters that encode an integer (one letter per digit)."""
    return "".join("aeioubcdfg"[int(d)] for d in f"{k:03d}")


def _person(k: int) -> str:
    return f"Fn{_word(k)} Ln{_word(k)}"


NON_PERSON = "Riverside Football Club"
"""An in-band sport subject that is not a plain personal name."""

ALL_ALIASES = {_person(9), _person(10)}
"""Occupation subjects whose aliases cover the whole answer pool."""


def person_table(per_relation: int = 40) -> pd.DataFrame:
    """People per person relation with popularity 1..per_relation."""
    rows = []
    for r, prop in enumerate(PERSON_RELATION_ORDER):
        objects = [f"{prop} answer {j}" for j in range(6)]
        for i in range(per_relation):
            k = r * per_relation + i
            name = NON_PERSON if (prop == "sport" and i == 9) else _person(k)
            obj = objects[i % 6]
            rows.append(
                {
                    "id": 1000 + k,
                    "subj": name,
                    "prop": prop,
                    "obj": obj,
                    "question": f"What is the {prop} of {name}?",
                    "possible_answers": objects if name in ALL_ALIASES else [obj],
                    "s_pop": i + 1,
                }
            )
    return pd.DataFrame(rows)


PARA_RELATIONS = ("occupation", "sport", "religion")


def paraphrase_table(per_relation: int = 10) -> pd.DataFrame:
    """Questions for three relations, plus an empty subject and a pool-wide alias."""
    rows = []
    for r, prop in enumerate(PARA_RELATIONS):
        objects = [f"{prop} value {j}" for j in range(6)]
        for i in range(per_relation + 1):
            qid = 100 * (r + 1) + i
            subj = "" if i == per_relation else f"Subject{qid}"
            obj = objects[i % 6]
            rows.append(
                {
                    "id": qid,
                    "subj": subj,
                    "prop": prop,
                    "obj": obj,
                    "question": f"What is {subj}'s {prop}?",
                    "possible_answers": objects if i == 0 else [obj],
                    "s_pop": 5 * i,
                }
            )
    return pd.DataFrame(rows)


def _records(df: pd.DataFrame, key: str) -> dict[Any, dict[str, Any]]:
    """Rows of a table keyed by one column (last row wins)."""
    return {r[key]: {str(c): v for c, v in r.items()} for r in df.to_dict("records")}


# --- a planted live backend ------------------------------------------------------

Plan = Mapping[str, tuple[str, float] | None]
"""Question text -> (chosen option, P(known)), or ``None`` to refuse."""


class PlannedBackend:
    """Answers each question as planned; a ``None`` plan is a refusal."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.calls = 0

    async def __aenter__(self) -> "PlannedBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def answer(self, request: Request, model: str) -> RawAnswer:
        self.calls += 1
        codes = list(request.questions["answer"]["criteria"])
        planned = self.plan[str(request.state["question"])]
        answers: dict[str, Any]
        if planned is None:
            answers = {
                "answer": {
                    "type": "refusal",
                    "question_type": "choice",
                    "options": codes,
                },
                "known": {"type": "refusal", "question_type": "noul", "options": []},
            }
        else:
            top, known = planned
            rest = 0.3 / (len(codes) - 1)
            answers = {
                "answer": {
                    "type": "choice",
                    "choice": top,
                    "probabilities": {c: 0.7 if c == top else rest for c in codes},
                },
                "known": {"type": "noul", "noul": known},
            }
        return RawAnswer(
            {
                "model": model,
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "answers": answers,
            }
        )


def _live(
    monkeypatch: pytest.MonkeyPatch, module: Any, plan: Plan
) -> Callable[[], PlannedBackend]:
    def factory() -> PlannedBackend:
        return PlannedBackend(plan)

    monkeypatch.setattr(module, "backend_for", lambda _s: factory)
    return factory


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        max_input_tokens=10**9,
        concurrency=4,
    )


# --- fresh names: item building --------------------------------------------------


def test_band_people_keeps_the_band_and_only_person_names() -> None:
    df = person_table()
    people = fn.band_people(df)
    assert set(people) == set(PERSON_RELATION_ORDER)
    for prop, names in people.items():
        g = _records(df[df["prop"] == prop], "subj")
        assert names == sorted(names)
        # 40 people with popularity 1..40: the (20 %, 40 %] band is 9..16.
        assert all(9 <= g[n]["s_pop"] <= 16 for n in names)
        assert all(is_person_name(n) for n in names)
    assert NON_PERSON not in people["sport"]
    assert len(people["sport"]) == 7
    assert len(people["occupation"]) == 8
    wide = fn.band_people(df, (0.0, 1.0))
    assert len(wide["father"]) == 39  # the minimum is not above its own quantile


def test_split_band_is_seeded_and_disjoint() -> None:
    people = {
        p: [f"{p.title().replace(' ', '')} N{c}" for c in "abcdef"]
        for p in PERSON_RELATION_ORDER
    }
    shared = "Shared Person"
    for p in PERSON_RELATION_ORDER:
        people[p] = sorted([*people[p], shared])
    donors, real = fn.split_band(people, seed=3)
    assert fn.split_band(people, seed=3) == (donors, real)
    assert fn.split_band(people, seed=4) != (donors, real)
    assert donors == sorted(donors)
    # A person in several relations who is drawn as a donor anywhere is never real.
    for p, names in real.items():
        assert not set(names) & set(donors)
        assert set(names) | (set(people[p]) & set(donors)) == set(people[p])
    assert shared in donors  # 7 people, 3 donors per relation: drawn somewhere
    none, everyone = fn.split_band(people, donor_share=0.0)
    assert none == []
    assert all(sorted(everyone[p]) == people[p] for p in people)


def test_unique_recombinations_uses_each_donor_once_each_way() -> None:
    donors = [_person(k) for k in range(10)]
    triples = fn.unique_recombinations(donors, random.Random(0), ())  # noqa: S311
    assert len(triples) == len(donors)
    assert sorted(a for _, a, _ in triples) == sorted(donors)
    assert sorted(b for _, _, b in triples) == sorted(donors)
    for name, a, b in triples:
        assert name == f"{a.split()[0]} {b.split()[1]}"
        assert a != b
    assert len({n for n, _, _ in triples}) == len(donors)
    again = fn.unique_recombinations(donors, random.Random(0), ())  # noqa: S311
    assert again == triples


def test_unique_recombinations_avoids_excluded_and_shared_first_names() -> None:
    a, b, c = "Ann Lee", "Bob Ray", "Cat Fox"
    # Three donors give two cycles; excluding one pair forces the other cycle.
    for seed in range(5):
        got = fn.unique_recombinations(
            [a, b, c],
            random.Random(seed),  # noqa: S311
            {"Ann Fox"},
        )
        assert {n for n, _, _ in got} == {"Ann Ray", "Bob Fox", "Cat Lee"}
    with pytest.raises(RuntimeError, match="no valid donor pairing"):
        fn.unique_recombinations([a, b], random.Random(0), {"Ann Ray"})  # noqa: S311
    with pytest.raises(RuntimeError, match="no valid donor pairing"):
        fn.unique_recombinations(
            ["Ann Lee", "Ann Ray"],
            random.Random(0),  # noqa: S311
            (),
            tries=5,
        )
    # Double surnames move together.
    two = fn.unique_recombinations(
        ["Ann Lee Park", "Bob Ray"],
        random.Random(0),  # noqa: S311
        (),
    )
    assert {n for n, _, _ in two} == {"Ann Ray", "Bob Lee Park"}


def _by_set(items: list[Item], name: str) -> list[Item]:
    return [it for it in items if it.info["set"] == name]


def test_fresh_items_matched_groups_and_never_real_names() -> None:
    df = person_table()
    items = fn.fresh_items(df)
    made, real = _by_set(items, "made_up"), _by_set(items, "real")
    assert items == made + real
    assert len(made) == len(real) > 0
    assert Counter(it.info["prop"] for it in made) == Counter(
        it.info["prop"] for it in real
    )
    subjects = set(df["subj"].astype(str))
    donors, reals = fn.split_band(fn.band_people(df))
    templates = {p: f"What is the {p} of {{}}?" for p in PERSON_RELATION_ORDER}
    pools = {p: {f"{p} answer {j}" for j in range(6)} for p in PERSON_RELATION_ORDER}
    for j, it in enumerate(it for it in made if it.info["prop"] == "occupation"):
        assert it.unit == f"fresh:made_up:occupation:{j}"
    for it in made:
        info = it.info
        assert info["subject"] not in subjects
        assert info["first_donor"] in donors
        assert info["surname_donor"] in donors
        assert info["subject"] == (
            f"{info['first_donor'].split()[0]} {info['surname_donor'].split()[1]}"
        )
        assert it.gold is None
        assert it.state["question"] == templates[info["prop"]].format(info["subject"])
        assert it.descriptions is not None
        assert set(it.descriptions.values()) <= pools[info["prop"]]
        assert len(set(it.descriptions.values())) == 4
        assert it.options == LETTERS[:4]
        assert it.nouls == {"known": KNOWN_QUESTION}
    # No token recurs: every first name and every surname is used at most once.
    assert len({it.info["first_donor"] for it in made}) == len(made)
    assert len({it.info["surname_donor"] for it in made}) == len(made)
    assert len({it.info["subject"] for it in made}) == len(made)
    row = _records(df, "subj")
    for it in real:
        subj = it.info["subject"]
        assert subj in reals[it.info["prop"]]
        assert subj not in donors
        assert subj not in ALL_ALIASES  # no distractors left: skipped
        assert it.gold is not None
        assert it.descriptions is not None
        assert it.descriptions[it.gold] == row[subj]["obj"]
        assert it.info["s_pop"] == float(row[subj]["s_pop"])
        assert isinstance(it.info["s_pop"], float)
    assert {it.info["subject"] for it in made}.isdisjoint(
        {it.info["subject"] for it in real}
    )


def test_fresh_items_are_seeded_and_respect_exclusions() -> None:
    df = person_table()
    first = fn.fresh_items(df)
    assert fn.fresh_items(df) == first
    other = fn.fresh_items(df, seed=1)
    assert other != first
    names = {it.info["subject"] for it in _by_set(first, "made_up")}
    again = fn.fresh_items(df, exclude=names)
    fresh = {it.info["subject"] for it in _by_set(again, "made_up")}
    assert fresh.isdisjoint(names)
    # The real group does not depend on the exclusions.
    assert _by_set(again, "real") == _by_set(first, "real")


def test_fresh_items_subsample_real_when_names_run_short() -> None:
    df = person_table()
    items = fn.fresh_items(df, donor_share=0.25)
    made, real = _by_set(items, "made_up"), _by_set(items, "real")
    donors, _ = fn.split_band(fn.band_people(df), donor_share=0.25)
    assert len(made) == len(real) <= len(donors)
    per = Counter(it.info["prop"] for it in made)
    assert per == Counter(it.info["prop"] for it in real)
    for it in real:  # the first ``kept`` real subjects of each relation
        assert int(it.unit.rsplit(":", 1)[1]) < per[it.info["prop"]]


# --- fresh names: experiment ------------------------------------------------------


def test_verdict_rules_and_precedence() -> None:
    def v(ci95: tuple[float, float], ci90: tuple[float, float]) -> str:
        return fx.verdict({"ci95": ci95, "ci90": ci90}, 0.05)

    assert v((0.01, 0.2), (0.02, 0.18)) == "known_better"
    assert v((-0.2, -0.01), (-0.18, -0.02)) == "known_worse"
    assert v((-0.06, 0.06), (-0.04, 0.04)) == "equivalent"
    assert v((-0.1, 0.1), (-0.08, 0.08)) == "inconclusive"
    # Margin bounds are strict.
    assert v((-0.06, 0.06), (-0.04, 0.05)) == "inconclusive"
    assert v((-0.06, 0.06), (-0.05, 0.04)) == "inconclusive"
    # A clear difference wins over a narrow interval inside the margin.
    assert v((0.01, 0.03), (0.015, 0.025)) == "known_better"
    assert v((-0.03, -0.01), (-0.025, -0.015)) == "known_worse"
    assert v((0.0, 0.03), (0.01, 0.02)) == "equivalent"
    assert fx.verdict({"ci95": (-0.06, 0.06), "ci90": (-0.04, 0.04)}, 0.03) == (
        "inconclusive"
    )


def _fresh_rows(
    items: list[Item],
    known: Callable[[bool], float],
    conf: Callable[[bool, int], float],
    refused: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows = []
    for k, it in enumerate(items):
        if it.unit in refused:
            continue
        made = it.info["set"] == "made_up"
        rows.append(
            {
                "unit": it.unit,
                **it.info,
                "p_known": known(made),
                "p_max": conf(made, k),
                "correct": None if made else k % 2 == 0,
            }
        )
    return rows


def _fake_embed(texts: list[str]) -> np.ndarray:
    return np.random.default_rng(len(texts)).normal(size=(len(texts), 4))


def test_fresh_analyse_planted_verdicts() -> None:
    items = fn.fresh_items(person_table())
    made = _by_set(items, "made_up")
    refused = frozenset({made[0].unit})
    cfg = fx.Config(bootstrap=30, permutations=0)
    refs = [_person(k) for k in range(240)]

    better = fx.analyse(
        _fresh_rows(items, lambda m: 0.1 if m else 0.9, lambda m, k: 0.6, refused),
        items,
        cfg,
        _fake_embed,
        refs,
    )
    p = better["primary"]
    assert p["verdict"] == "known_better"
    assert p["diff"] == pytest.approx(0.5)
    assert p["margin"] == 0.05
    assert better["auroc"]["one_minus_known"]["auroc"] == 1.0
    assert better["auroc"]["one_minus_p_max"]["auroc"] == 0.5
    n = better["n"]
    assert n["made_up"] == len(made) - 1
    assert n["real"] == len(items) - len(made)
    assert n["refused_share"]["made_up"] == pytest.approx(1 / len(made))
    assert n["refused_share"]["real"] == 0.0
    # Made-up names cluster by first-name donor, real people by themselves.
    assert n["first_name_clusters"] == len(items) - 1
    assert set(better["secondary"]["leave_one_relation_out"]) == set(
        PERSON_RELATION_ORDER
    )
    assert all(
        d == pytest.approx(0.5)
        for d in better["secondary"]["leave_one_relation_out"].values()
    )
    assert better["secondary"]["by_relation"]["diff"] == pytest.approx(0.5)
    text = better["secondary"]["text_only_folds_by_subject"]
    assert {"char_ngram", "word_ngram", "name_structure"} <= set(text)
    d = better["descriptive"]
    assert d["mean_p_known"] == {
        "made_up": pytest.approx(0.1),
        "real": pytest.approx(0.9),
    }
    assert d["real_accuracy"] == pytest.approx(
        np.mean([k % 2 == 0 for k, it in enumerate(items) if it.info["set"] == "real"])
    )
    assert all(v == 1.0 for v in d["auroc_known_by_relation"].values())

    worse = fx.analyse(
        _fresh_rows(items, lambda m: 0.5, lambda m, k: 0.2 if m else 0.8),
        items,
        cfg,
        _fake_embed,
        refs,
    )
    assert worse["primary"]["verdict"] == "known_worse"
    assert worse["primary"]["diff"] == pytest.approx(-0.5)
    assert worse["n"]["refused_share"] == {"made_up": 0.0, "real": 0.0}

    same = fx.analyse(
        _fresh_rows(items, lambda m: 0.3 if m else 0.7, lambda m, k: 0.3 if m else 0.7),
        items,
        cfg,
        _fake_embed,
        refs,
    )
    assert same["primary"]["diff"] == 0.0
    assert same["primary"]["ci90"] == pytest.approx([0.0, 0.0])
    assert same["primary"]["verdict"] == "equivalent"


def test_fresh_build_items_excludes_earlier_lookalikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from beyond_answer_confidence.tasks import shortcuts

    df = person_table()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: df)
    cfg = fx.Config(lookalike_per_relation=3, replicates=2)
    items = fx.build_items(_settings(tmp_path), cfg)
    rng = random.Random(f"{shortcuts._SEED_NAMESPACE}:lookalike")  # noqa: S311
    earlier = {
        n
        for picks in shortcuts.lookalike_subjects(df, 3, rng).values()
        for n, _, _ in picks
    }
    assert items == fn.fresh_items(df, earlier)
    assert {it.info["subject"] for it in items}.isdisjoint(earlier)
    tasks = fx.requests(_settings(tmp_path), cfg)
    assert len(tasks) == 2 * len(items)
    assert fx.EXPERIMENT.name == "fresh_names"


def test_fresh_run_offline_without_cache_skips_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: person_table())
    s = _settings(tmp_path)
    summary = fx.run(s, fx.Config(replicates=1, lookalike_per_relation=2))
    assert "results" not in summary
    assert summary["complete_items"] == 0
    assert (s.out(fx.NAME) / "summary.json").exists()


def test_fresh_run_with_planted_backend_and_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = person_table()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: df)
    monkeypatch.setattr(
        distractors, "embed", lambda texts, _d: _fake_embed(list(texts))
    )
    s = _settings(tmp_path)
    cfg = fx.Config(
        replicates=1, lookalike_per_relation=2, bootstrap=20, permutations=1
    )
    items = fx.build_items(s, cfg)
    refused = next(it for it in items if it.info["set"] == "made_up")
    plan: dict[str, tuple[str, float] | None] = {
        str(it.state["question"]): (
            "A",
            0.1 if it.info["set"] == "made_up" else 0.9,
        )
        for it in items
    }
    plan[str(refused.state["question"])] = None
    _live(monkeypatch, fx, plan)
    summary = fx.run(s, cfg)
    assert summary["errors"] == {"RefusalError": 1}
    assert summary["complete_items"] == len(items) - 1
    res = summary["results"]
    assert res["primary"]["verdict"] == "known_better"
    assert res["n"]["refused_share"]["made_up"] > 0
    perm = res["secondary"]["text_only_folds_by_subject"]["char_ngram"]["permutation"]
    assert perm["shuffles"] == 1
    assert (s.out(fx.NAME) / "rows.jsonl").exists()


# --- paraphrases: items -----------------------------------------------------------


def test_paraphrase_wordings_are_distinct_templates() -> None:
    assert len(pp.PARAPHRASES) == 16
    for wordings in pp.PARAPHRASES.values():
        assert len(wordings) == 4
        assert len(set(wordings)) == 4
        assert all(w.count("{}") == 1 for w in wordings)
        assert len({w.format("X") for w in wordings}) == 4


def test_paraphrase_items_share_options_across_wordings() -> None:
    df = paraphrase_table()
    items = pp.paraphrase_items(df, per_relation=4)
    assert pp.paraphrase_items(df, per_relation=4) == items
    assert pp.paraphrase_items(df, per_relation=4, seed=1) != items
    by_q: dict[int, list[Item]] = {}
    for it in items:
        by_q.setdefault(it.info["question_id"], []).append(it)
    assert len(by_q) == 4 * len(PARA_RELATIONS)
    row = _records(df, "id")
    for qid, group in by_q.items():
        assert [it.info["wording"] for it in group] == [0, 1, 2, 3]
        assert [it.unit for it in group] == [f"para:{qid}:{k}" for k in range(4)]
        prop = group[0].info["prop"]
        subj = str(row[qid]["subj"])
        assert subj != ""
        assert row[qid]["possible_answers"] == [
            row[qid]["obj"]
        ]  # pool-wide alias skipped
        questions = [it.state["question"] for it in group]
        assert questions == [w.format(subj) for w in pp.PARAPHRASES[prop]]
        assert len(set(questions)) == 4
        first = group[0]
        assert first.descriptions is not None
        assert first.gold is not None
        assert first.descriptions[first.gold] == row[qid]["obj"]
        assert len(set(first.descriptions.values())) == 4
        for it in group[1:]:
            assert it.options == first.options
            assert it.descriptions == first.descriptions
            assert it.gold == first.gold
        assert first.info["s_pop"] == float(row[qid]["s_pop"])
        assert first.nouls == {"known": KNOWN_QUESTION}
    # Asking for more than a relation has takes every usable question.
    every = pp.paraphrase_items(df, per_relation=100)
    assert len(every) == 4 * 9 * len(PARA_RELATIONS)


# --- paraphrases: analysis ----------------------------------------------------------


def _para_rows() -> list[dict[str, Any]]:
    """Forty questions of four planted kinds (by ``q % 4``).

    0: stable and reliable; 1: stable, always wrong; 2: alternates right and
    wrong; 3: right three times then wrong. P(known) is 0.9 only for kind 0;
    p_max rises with the wording.
    """
    rows: list[dict[str, Any]] = []
    for q in range(40):
        kind = q % 4
        tops = {
            0: "AAAA",
            1: "BBBB",
            2: "ABAB",
            3: "AAAC",
        }[kind]
        rows.extend(
            {
                "question_id": q,
                "subject": f"S{q}",
                "prop": PARA_RELATIONS[q % 3],
                "wording": k,
                "top": tops[k],
                "correct": tops[k] == "A",
                "p_known": 0.9 if kind == 0 else 0.2,
                "p_max": 0.5 + 0.1 * k,
            }
            for k in range(4)
        )
    # A question with only three answered wordings (one refused).
    rows += [
        {
            "question_id": 99,
            "subject": "S99",
            "prop": "sport",
            "wording": k,
            "top": "A",
            "correct": True,
            "p_known": 0.5,
            "p_max": 0.5,
        }
        for k in range(3)
    ]
    return rows


def test_paraphrase_analyse_planted_answers() -> None:
    res = px.analyse(_para_rows(), px.Config(bootstrap=30))
    assert res["n"] == {
        "questions": 40,
        "questions_dropped_incomplete": 1,
        "answers": 160,
        "subjects": 40,
    }
    assert res["accuracy"] == pytest.approx(9 / 16)
    assert res["accuracy_by_wording"] == {
        0: pytest.approx(0.75),
        1: pytest.approx(0.5),
        2: pytest.approx(0.75),
        3: pytest.approx(0.25),
    }
    stab, rel = res["stability"], res["reliability"]
    assert stab["share"] == pytest.approx(0.5)
    assert rel["share"] == pytest.approx(0.25)
    assert rel["auroc_mean_known"]["auroc"] == 1.0
    # Mean p_max is the same for every question: no information.
    assert rel["auroc_mean_p_max"]["auroc"] == 0.5
    assert rel["known_minus_p_max"]["diff"] == pytest.approx(0.5)
    lo, hi = rel["leave_one_relation_out_range"]
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)
    # Stable = kinds 0 and 1; only kind 0 has high P(known).
    assert stab["auroc_mean_known"]["auroc"] == pytest.approx(0.75)
    corr = res["correctness"]
    assert 0.5 < corr["auroc_known"]["auroc"] <= 1.0
    assert corr["known_minus_p_max"]["diff"] == pytest.approx(
        corr["auroc_known"]["auroc"] - corr["auroc_p_max"]["auroc"]
    )
    beyond = res["beyond_p_max"]
    assert set(beyond) == {"p_max_only", "p_max_and_known", "difference"}
    assert beyond["p_max_and_known"]["auroc"] > beyond["p_max_only"]["auroc"]
    assert res["consistency"]["mean_sd_known"] == 0.0
    assert res["consistency"]["mean_sd_p_max"] == pytest.approx(
        float(np.std([0.5, 0.6, 0.7, 0.8], ddof=1))
    )
    assert px.analyse(_para_rows(), px.Config(bootstrap=30)) == res


def test_logit_clips_extremes() -> None:
    out = px._logit(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(out))
    assert out[1] == 0.0
    assert out[0] == pytest.approx(-out[2])
    assert out[2] == pytest.approx(np.log((1 - px.EPS) / px.EPS))


def test_paraphrase_requests_and_offline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = paraphrase_table()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: df)
    s = _settings(tmp_path)
    cfg = px.Config(per_relation=3, replicates=2)
    assert px.build_items(s, cfg) == pp.paraphrase_items(df, 3)
    assert len(px.requests(s, cfg)) == 2 * 4 * 3 * len(PARA_RELATIONS)
    summary = px.run(s, cfg)
    assert "results" not in summary
    assert summary["complete_items"] == 0
    assert px.EXPERIMENT.name == "paraphrase_stability"


def test_paraphrase_run_with_planted_backend_drops_refused_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = paraphrase_table()
    monkeypatch.setattr(loaders, "load_popqa", lambda _d: df)
    s = _settings(tmp_path)
    cfg = px.Config(per_relation=8, bootstrap=20)
    items = px.build_items(s, cfg)
    plan: dict[str, tuple[str, float] | None] = {}
    seen: Counter[str] = Counter()
    kinds: dict[int, int] = {}
    for it in items:
        qid, k = it.info["question_id"], it.info["wording"]
        if qid not in kinds:
            kinds[qid] = seen[it.info["prop"]] % 4
            seen[it.info["prop"]] += 1
        gold = str(it.gold)
        wrong = next(o for o in it.options if o != gold)
        pattern = ("gggg", "wwww", "gwgw", "gggw")[kinds[qid]]
        top = gold if pattern[k] == "g" else wrong
        plan[str(it.state["question"])] = (top, 0.9 if kinds[qid] == 0 else 0.2)
    refused = next(it for it in items if kinds[it.info["question_id"]] == 3)
    plan[str(refused.state["question"])] = None
    _live(monkeypatch, px, plan)
    summary = px.run(s, cfg)
    assert summary["errors"] == {"RefusalError": 1}
    res = summary["results"]
    assert res["n"]["questions"] == 8 * len(PARA_RELATIONS) - 1
    assert res["n"]["questions_dropped_incomplete"] == 1
    assert res["reliability"]["auroc_mean_known"]["auroc"] == 1.0
    assert res["stability"]["share"] == pytest.approx(12 / 23)
    assert (s.out(px.NAME) / "summary.json").exists()
