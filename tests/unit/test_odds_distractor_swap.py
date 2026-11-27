import itertools
import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from beyond_answer_confidence.backends.fake import FakeBackend
from beyond_answer_confidence.data import distractors as dis_mod
from beyond_answer_confidence.data import loaders
from beyond_answer_confidence.data.distractors import normalise_answer
from beyond_answer_confidence.experiments import distractor_swap as dsx
from beyond_answer_confidence.experiments import odds_semantics as osx
from beyond_answer_confidence.metrics.distributions import tv
from beyond_answer_confidence.settings import Settings
from beyond_answer_confidence.tasks import distractor_swap as dst
from beyond_answer_confidence.tasks import odds_semantics as ost
from beyond_answer_confidence.tasks.benchmarks import display_case
from beyond_answer_confidence.tasks.chance import COLOURS, DEVICES, INSTRUCTIONS
from beyond_answer_confidence.tasks.schema import Item, order_collisions

# Stated odds: items.


def test_nth_permutation_matches_itertools_order() -> None:
    labels = list("abcde")
    every = list(itertools.permutations(labels))
    assert [ost._nth_permutation(labels, i) for i in range(len(every))] == every


def test_orders_match_sampling_the_full_list() -> None:
    for k in range(1, 8):
        labels = [f"o{i}" for i in range(k)]
        perms = list(itertools.permutations(labels))
        for n in (1, 3, 6, 50):
            rng = random.Random(f"{ost.SEED_NAMESPACE}:orders:dev{k}")  # noqa: S311
            expected = perms if len(perms) <= n else rng.sample(perms, n)
            assert ost._orders(labels, n, f"dev{k}") == expected


def test_orders_of_eleven_outcomes_are_sampled_without_listing_them() -> None:
    labels = [str(v) for v in range(2, 13)]
    orders = ost._orders(labels, 4, "two_dice_sum")
    assert len(set(orders)) == 4
    assert all(sorted(o) == sorted(labels) for o in orders)


def test_spinner_profiles_are_distributions_in_steps_of_five() -> None:
    profiles = ost.spinner_profiles(8, seed=3)
    assert [len(p) for p in profiles] == [3, 4, 5, 6, 3, 4, 5, 6]
    for p in profiles:
        assert sum(p.values()) == pytest.approx(1.0)
        assert set(p) <= set(COLOURS)
        assert all(v >= 0.05 for v in p.values())
        assert all(round(v * 20) == pytest.approx(v * 20) for v in p.values())
    assert profiles == ost.spinner_profiles(8, seed=3)
    assert profiles != ost.spinner_profiles(8, seed=4)


def test_spinner_state_lists_every_sector() -> None:
    text = ost.spinner_state({"red": 0.5, "blue": 0.25, "green": 0.25})
    assert text == (
        "A spinner is divided into coloured sectors: red covers 50%, blue covers "
        "25% and green covers 25% of it. It is spun once."
    )


def test_orders_are_distinct_permutations() -> None:
    labels = ["a", "b", "c", "d"]
    every = ost._orders(labels, 100, "k")
    assert len(every) == math.factorial(4)
    some = ost._orders(labels, 5, "k")
    assert len(some) == len(set(some)) == 5
    assert all(sorted(o) == labels for o in some)
    assert some == ost._orders(labels, 5, "k")


def _odds_items() -> list[Item]:
    return ost.odds_items(device_orders=2, spinners=4, spinner_orders=3, seed=0)


def test_odds_items_counts_and_options() -> None:
    items = _odds_items()
    by_set: dict[str, list[Item]] = {}
    for it in items:
        by_set.setdefault(it.info["set"], []).append(it)
    assert len(by_set["device"]) == len(DEVICES) * 2 * 2
    assert len(by_set["spinner"]) == 4 * 3 * 2
    assert len(by_set["spinner_yes_no"]) == 4
    assert len({it.unit for it in items}) == len(items)
    for it in by_set["device"] + by_set["spinner"]:
        ideal = it.info["ideal"]
        assert sum(ideal.values()) == pytest.approx(1.0)
        assert sorted(it.options) == sorted(ideal)
        assert list(it.options) == it.info["order"]
        assert it.instructions == ost.WORDINGS[it.info["arm"]]
    for it in by_set["spinner_yes_no"]:
        assert not it.options
        assert it.unit.endswith(":0")
        assert set(it.nouls) == {f"lands_{c}" for c in it.info["ideal"]}
        assert it.info["arm"] == "yes_no"
    # Each spinner's option orders within one wording are distinct.
    orders: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for it in by_set["spinner"]:
        orders.setdefault((it.info["key"], it.info["arm"]), []).append(it.options)
    assert all(len(set(v)) == len(v) == 3 for v in orders.values())


def test_log_score_and_plain_differ_only_in_instructions() -> None:
    items = _odds_items()
    plain = {
        it.unit.replace(":plain:", ":"): it for it in items if it.info["arm"] == "plain"
    }
    logs = {
        it.unit.replace(":log_score:", ":"): it
        for it in items
        if it.info["arm"] == "log_score"
    }
    assert set(plain) == set(logs)
    for u, p in plain.items():
        q = logs[u]
        assert p.instructions == INSTRUCTIONS
        assert q.instructions == ost.LOG_SCORE
        assert p.state == q.state
        assert p.options == q.options
        assert {**p.info, "arm": "x"} == {**q.info, "arm": "x"}
        rp, rq = p.request(0).questions, q.request(0).questions
        assert rp.keys() == rq.keys() == {"answer"}
        assert {**rp["answer"], "instructions": ""} == {
            **rq["answer"],
            "instructions": "",
        }


def test_build_gives_each_order_its_own_replicate() -> None:
    items, tasks = osx.build(osx.Config(device_orders=3, spinners=2, spinner_orders=2))
    assert len(tasks) == len(items)
    assert order_collisions(tasks, "m") == 0
    reps = {t.unit: t.request.replicate for t in tasks}
    assert reps["odds:device:plain:die_words:2"] == 2
    assert reps["odds:spinner_yes_no:1:0"] == 0
    assert (
        osx.requests(Settings(), osx.Config(spinners=1))
        == osx.build(osx.Config(spinners=1))[1]
    )


# Stated odds: analysis.


def _rows(
    dists: Sequence[dict[str, float]],
    orders: Sequence[Sequence[str]],
    ideal: dict[str, float],
    key: str = "k",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dist": list(dists),
            "order": [list(o) for o in orders],
            "ideal": [ideal] * len(dists),
            "key": [key] * len(dists),
        }
    )


def test_choice_metrics_answer_equal_to_stated_odds() -> None:
    ideal = {"a": 0.5, "b": 0.3, "c": 0.2}
    g = _rows([ideal, ideal], [("a", "b", "c"), ("c", "b", "a")], ideal)
    m = osx.choice_metrics(g)
    assert m["n_calls"] == 2
    assert m["single_call_tv"] == pytest.approx(0.0)
    assert m["order_mean_tv"] == pytest.approx(0.0)
    assert m["mean_abs_error"] == pytest.approx(0.0)
    assert m["first_listed_mass"] == pytest.approx(0.35)
    assert m["first_listed_excess"] == pytest.approx(0.0)
    assert m["mode_mass"] == m["mode_stated"] == pytest.approx(0.5)


def test_choice_metrics_uniform_answer() -> None:
    ideal = {"a": 0.5, "b": 0.3, "c": 0.2}
    uniform = dict.fromkeys(ideal, 1 / 3)
    g = _rows([uniform], [("c", "a", "b")], ideal)
    m = osx.choice_metrics(g)
    assert m["single_call_tv"] == pytest.approx(1 / 6)
    assert m["order_mean_tv"] == pytest.approx(1 / 6)
    assert m["mean_abs_error"] == pytest.approx((1 / 6 + 1 / 30 + 2 / 15) / 3)
    assert m["first_listed_mass"] == pytest.approx(1 / 3)
    assert m["first_listed_excess"] == pytest.approx(1 / 3 - 0.2)
    assert m["mode_mass"] == pytest.approx(1 / 3)
    assert m["mode_stated"] == pytest.approx(0.5)


def test_choice_metrics_first_listed_bias_averages_out_over_orders() -> None:
    ideal = {"a": 0.5, "b": 0.5}
    g = _rows(
        [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], [("a", "b"), ("b", "a")], ideal
    )
    m = osx.choice_metrics(g)
    assert m["single_call_tv"] == pytest.approx(0.5)
    assert m["order_mean_tv"] == pytest.approx(0.0)
    assert m["first_listed_mass"] == pytest.approx(1.0)
    assert m["first_listed_excess"] == pytest.approx(0.5)
    # Two keys: one averaged per key, not pooled.
    other = _rows([{"a": 1.0, "b": 0.0}], [("a", "b")], ideal, key="k2")
    assert osx.choice_metrics(pd.concat([g, other]))["order_mean_tv"] == (
        pytest.approx(0.25)
    )


def _planted_rows(items: Sequence[Item], scale: float = 1.0) -> list[dict[str, Any]]:
    """Log-score answers equal the stated odds; plain answers are uniform."""
    rows: list[dict[str, Any]] = []
    for it in items:
        ideal = it.info["ideal"]
        row: dict[str, Any] = {**it.info, "unit": it.unit}
        if it.info["set"] == "spinner_yes_no":
            row |= {f"p_lands_{c}": scale * p for c, p in ideal.items()}
        elif it.info["arm"] == "log_score":
            row["dist"] = dict(ideal)
        else:
            row["dist"] = dict.fromkeys(ideal, 1 / len(ideal))
        rows.append(row)
    return rows


def test_analyse_planted_answers() -> None:
    items = _odds_items()
    res = osx.analyse(_planted_rows(items), osx.Config(bootstrap=30))
    assert set(res) == {"device", "spinner", "spinner_yes_no"}
    for part in ("device", "spinner"):
        block = res[part]
        assert block["log_score"]["single_call_tv"] == pytest.approx(0.0)
        assert block["log_score"]["first_listed_excess"] == pytest.approx(0.0)
        keyed: dict[str, dict[str, float]] = {
            it.info["key"]: it.info["ideal"] for it in items if it.info["set"] == part
        }
        uni_tv = [tv(dict.fromkeys(i, 1 / len(i)), i) for i in keyed.values()]
        expected = sum(uni_tv) / len(uni_tv)
        effect = block["log_score_minus_plain_single_call_tv"]
        assert effect["n_keys"] == len(keyed)
        assert effect["mean"] == pytest.approx(-expected)
        lo, hi = effect["ci95"]
        assert lo <= effect["mean"] <= hi <= 0.0
        assert block["plain"]["n_calls"] == block["log_score"]["n_calls"]
    # Every device but the loaded die is uniform.
    assert res["device"]["plain"]["single_call_tv"] > 0.0
    yn = res["spinner_yes_no"]
    assert yn["n_profiles"] == 4
    assert yn["mean_abs_error"] == pytest.approx(0.0)
    assert yn["mean_sum_of_p_yes"] == pytest.approx(1.0)
    assert yn["mean_abs_error_normalised"] == pytest.approx(0.0)
    # Doubled yes/no answers: sum 2, normalised error still 0.
    doubled = osx.analyse(_planted_rows(items, 2.0), osx.Config(bootstrap=5))
    assert doubled["spinner_yes_no"]["mean_sum_of_p_yes"] == pytest.approx(2.0)
    assert doubled["spinner_yes_no"]["mean_abs_error_normalised"] == (
        pytest.approx(0.0)
    )
    assert doubled["spinner_yes_no"]["mean_abs_error"] > 0.0
    assert osx.analyse(_planted_rows(items), osx.Config(bootstrap=30)) == res


def _live(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    def factory(settings: Settings) -> Callable[[], FakeBackend] | None:
        return FakeBackend if settings.live else None

    monkeypatch.setattr(module, "backend_for", factory)


def test_odds_run_with_fake_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = osx.Config(device_orders=1, spinners=2, spinner_orders=1, bootstrap=5)
    offline = Settings(data_dir=tmp_path / "d", output_dir=tmp_path / "o")
    empty = osx.run(offline, cfg)
    assert "results" not in empty
    assert empty["complete_items"] == 0
    _live(monkeypatch, osx)
    live = Settings(
        data_dir=tmp_path / "d",
        output_dir=tmp_path / "o",
        live=True,
        max_input_tokens=10**9,
    )
    summary = osx.run(live, cfg)
    assert summary["complete_items"] == summary["items"] == len(osx.build(cfg)[0])
    res = summary["results"]
    # The fake puts 0.5 + 0.5 / k on the first option and 0.5 / k on the rest.
    words = [it for it in osx.build(cfg)[0] if it.info["key"] == "die_words"]
    assert len(words) == 2
    first = res["device"]["plain"]["first_listed_mass"]
    assert 0.5 < first < 1.0
    yn = res["spinner_yes_no"]
    assert yn["n_profiles"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "o" / osx.NAME / "rows.jsonl").read_text().splitlines()
    ]
    die = next(r for r in rows if r["unit"] == "odds:device:plain:die_words:0")
    assert die["dist"][die["order"][0]] == pytest.approx(0.5 + 0.5 / 6)
    assert tv(die["dist"], die["ideal"]) == pytest.approx(0.5 - 0.5 / 6)
    spin = next(r for r in rows if r["set"] == "spinner_yes_no")
    for c in spin["ideal"]:
        assert spin[f"p_lands_{c}"] == 0.25 + 0.5 * (len(f"lands_{c}") % 2)
    saved = json.loads((tmp_path / "o" / osx.NAME / "summary.json").read_text())
    assert saved["results"]["spinner_yes_no"] == pytest.approx(yn)
    # The second offline run answers from the cache.
    again = osx.run(offline, cfg)
    assert repr(again["results"]) == repr(res)


# Distractor swap: items.


def _picker(
    skip: frozenset[str] = frozenset(), *, use_aliases: bool = True
) -> Callable[[Sequence[str], Sequence[Sequence[str]], Sequence[str]], Any]:
    """Nearest = the first three pool answers that are not the gold or its aliases."""

    def pick(
        golds: Sequence[str], aliases: Sequence[Sequence[str]], pool: Sequence[str]
    ) -> list[list[str] | None]:
        out: list[list[str] | None] = []
        for g, al in zip(golds, aliases, strict=True):
            banned = {normalise_answer(a) for a in (g, *(al if use_aliases else []))}
            cand = [p for p in dict.fromkeys(pool) if normalise_answer(p) not in banned]
            out.append(None if g in skip or len(cand) < 3 else cand[-3:])
        return out

    return pick


CITIES = ["PARIS", "Lyon", "Paris, France", "Nice", "Lille", "Rome", "Milan"]


def triviaqa_table() -> pd.DataFrame:
    rows = []
    for i, c in enumerate(CITIES):
        aliases = ["Paris, France", "Paree"] if c == "PARIS" else [c.lower()]
        ans = {"value": c, "aliases": aliases, "type": "City"}
        rows.append({"question_id": f"tq{i}", "question": f"Q{i}?", "answer": ans})
    # A repeated question (one row per evidence document).
    rows.append(dict(rows[0]))
    # Too few golds of this type for a pool draw.
    for i in range(2):
        ans = {"value": f"{i + 1}", "aliases": [], "type": "Number"}
        rows.append({"question_id": f"num{i}", "question": "N?", "answer": ans})
    return pd.DataFrame(rows)


def _check_swap_items(
    items: Sequence[Item], golds: dict[str, tuple[str, list[str]]]
) -> None:
    pairs: dict[str, set[str]] = {}
    for it in items:
        assert it.descriptions is not None
        texts = [it.descriptions[o] for o in it.options]
        gold_text, aliases = golds[it.info["question_id"]]
        assert texts.count(gold_text) == 1
        assert it.gold is not None
        assert it.descriptions[it.gold] == gold_text
        banned = {normalise_answer(a) for a in (gold_text, *aliases)}
        others = [t for o, t in it.descriptions.items() if o != it.gold]
        assert len(others) == len(set(others)) == 3
        assert not banned & {normalise_answer(t) for t in others}
        assert it.unit == (
            f"swap:{it.info['set']}:{it.info['arm']}:{it.info['question_id']}"
        )
        assert set(it.nouls) == {"known"}
        pairs.setdefault(it.info["question_id"], set()).add(it.info["arm"])
    assert all(v == {"nearest", "pool"} for v in pairs.values())


def test_triviaqa_swap_items() -> None:
    table = triviaqa_table()
    items = dst.triviaqa_swap_items(table, _picker(frozenset({"Rome"})), n=10, seed=0)
    golds = {
        r["question_id"]: (display_case(r["answer"]["value"]),
                           list(r["answer"]["aliases"]))
        for r in table.to_dict("records")
    }  # fmt: skip
    _check_swap_items(items, golds)
    qids = {it.info["question_id"] for it in items}
    # Rome has no nearest options; the numbers have no pool.
    assert qids == {f"tq{i}" for i in range(len(CITIES))} - {"tq5"}
    assert all(it.info["answer_type"] == "City" for it in items)
    paris = next(it for it in items if it.unit == "swap:triviaqa:pool:tq0")
    assert paris.descriptions is not None
    assert "Paris" in paris.descriptions.values()
    assert "Paris, France" not in paris.descriptions.values()
    # The arms differ in their wrong options, never in the question.
    near = next(it for it in items if it.unit == "swap:triviaqa:nearest:tq0")
    assert near.state == paris.state
    capped = dst.triviaqa_swap_items(table, _picker(), n=2, seed=0)
    assert len(capped) == 4
    assert capped == dst.triviaqa_swap_items(table, _picker(), n=2, seed=0)


def popqa_table() -> pd.DataFrame:
    rows = []
    for i in range(12):
        prop = "genre" if i < 8 else "sport"
        rows.append(
            {
                "id": 100 + i,
                "question": f"What is the {prop} of Thing{i}?",
                "obj": f"{prop}-value-{i % 6}",
                "prop": prop,
                "possible_answers": [f"{prop}-value-{i % 6}"],
                "o_aliases": json.dumps([f"{prop}-value-{(i + 1) % 6}"])
                if i % 2
                else "",
                "s_pop": str(10 * i),
            }
        )
    return pd.DataFrame(rows)


def test_popqa_swap_items() -> None:
    table = popqa_table()
    items = dst.popqa_swap_items(table, _picker(), n=50, seed=1)
    golds = {
        str(r["id"]): (
            str(r["obj"]),
            list(r["possible_answers"]) + json.loads(str(r["o_aliases"]) or "[]"),
        )
        for r in table.to_dict("records")
    }
    _check_swap_items(items, golds)
    # Sport has four objects: a sport alias in the pool leaves only two.
    qids = {it.info["question_id"] for it in items}
    assert {"109", "111"} - qids == {"109"}
    for it in items:
        assert it.info["s_pop"] == float(int(it.info["question_id"]) - 100) * 10
    assert len(dst.popqa_swap_items(table, _picker(), n=1, seed=1)) == 2
    assert dst.popqa_swap_items(table, _picker(), n=50, seed=1) == items


def test_popqa_skips_questions_without_nearest_options() -> None:
    table = popqa_table()
    # Nearest options that ignore aliases, but no pool draw for 109.
    loose = _picker(use_aliases=False)
    qids = {it.info["question_id"] for it in dst.popqa_swap_items(table, loose)}
    assert "109" not in qids
    assert "111" in qids
    skip = frozenset(f"genre-value-{i}" for i in range(6))
    items = dst.popqa_swap_items(table, _picker(skip), n=50, seed=0)
    assert all(it.info["prop"] == "sport" for it in items)


# Distractor swap: analysis.


def test_arm_metrics_planted() -> None:
    g = pd.DataFrame(
        {
            "p_max": [0.9, 0.8, 0.6, 0.4],
            "correct": [True, True, False, False],
            "p_known": [0.7, 0.6, 0.5, 0.4],
        }
    )
    m = dsx.arm_metrics(g)
    assert m["n"] == 4
    assert m["accuracy"] == 0.5
    assert m["mean_p_max"] == pytest.approx(0.675)
    assert m["error_auroc_p_max"] == 1.0
    assert m["error_auroc_known"] == 1.0
    assert m["selective_accuracy_50"] == 1.0
    assert m["selective_accuracy_80"] == pytest.approx(2 / 3)
    assert 0.0 <= m["smece"] <= 1.0


def _swap_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("triviaqa", "popqa"):
        for q in range(10):
            orig = dsx.ORIGINAL[name]
            other = "pool" if orig == "nearest" else "nearest"
            rows.append(
                {
                    "set": name,
                    "question_id": f"q{q}",
                    "arm": orig,
                    "correct": True,
                    "p_max": 0.9,
                    "p_known": 0.8,
                }
            )
            rows.append(
                {
                    "set": name,
                    "question_id": f"q{q}",
                    "arm": other,
                    "correct": q % 2 == 0,
                    "p_max": 0.6,
                    "p_known": 0.4,
                }
            )
        # A question asked in one arm only is left out.
        rows.append(
            {
                "set": name,
                "question_id": "solo",
                "arm": "pool",
                "correct": False,
                "p_max": 0.1,
                "p_known": 0.1,
            }
        )
    return rows  # fmt: skip


def test_swap_analyse_planted() -> None:
    res = dsx.analyse(_swap_rows(), dsx.Config(bootstrap=40))
    assert set(res) == {"triviaqa", "popqa"}
    for name in ("triviaqa", "popqa"):
        block = res[name]
        orig = dsx.ORIGINAL[name]
        other = "pool" if orig == "nearest" else "nearest"
        assert block["original_arm"] == orig
        assert block["arms"][orig]["accuracy"] == 1.0
        assert block["arms"][orig]["n"] == 10
        assert block["arms"][other]["accuracy"] == 0.5
        assert block["arms"][other]["mean_p_max"] == pytest.approx(0.6)
        diff = block["other_minus_original"]
        assert diff["n_questions"] == 10
        assert diff["accuracy"] == pytest.approx(-0.5)
        assert diff["mean_p_max"] == pytest.approx(-0.3)
        assert diff["mean_p_max_ci95"] == pytest.approx((-0.3, -0.3))
        lo, hi = diff["accuracy_ci95"]
        assert -1.0 <= lo <= -0.5 <= hi <= 0.0
    # The original arm makes no errors, so its error AUROCs are undefined.
    assert math.isnan(res["triviaqa"]["arms"]["nearest"]["error_auroc_p_max"])
    assert repr(dsx.analyse(_swap_rows(), dsx.Config(bootstrap=40))) == repr(res)


def _fake_data(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    seen: list[Path] = []

    def picker(data_dir: Path, k: int = 3) -> Any:
        seen.append(data_dir)
        return _picker()

    monkeypatch.setattr(dis_mod, "distractor_picker", picker)
    monkeypatch.setattr(loaders, "load_triviaqa_validation", lambda d: triviaqa_table())
    monkeypatch.setattr(loaders, "load_popqa", lambda d: popqa_table())
    return seen


def test_swap_build_items_and_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_data(monkeypatch)
    settings = Settings(data_dir=tmp_path)
    cfg = dsx.Config(per_set=3, replicates=2)
    items = dsx.build_items(settings, cfg)
    assert seen == [tmp_path]
    sets = [it.info["set"] for it in items]
    assert sets.count("triviaqa") == sets.count("popqa") == 6
    tasks = dsx.requests(settings, cfg)
    assert len(tasks) == 2 * len(items)
    assert {t.request.replicate for t in tasks} == {0, 1}


def test_swap_run_with_fake_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_data(monkeypatch)
    cfg = dsx.Config(per_set=4, bootstrap=5)
    offline = Settings(data_dir=tmp_path / "d", output_dir=tmp_path / "o")
    assert "results" not in dsx.run(offline, cfg)
    _live(monkeypatch, dsx)
    live = Settings(
        data_dir=tmp_path / "d",
        output_dir=tmp_path / "o",
        live=True,
        max_input_tokens=10**9,
    )
    summary = dsx.run(live, cfg)
    assert summary["complete_items"] == summary["items"] == 16
    rows = [
        json.loads(line)
        for line in (tmp_path / "o" / dsx.NAME / "rows.jsonl").read_text().splitlines()
    ]
    # The fake always picks option A and answers "known" with 0.75.
    assert all(r["top"] == "A" for r in rows)
    assert all(r["correct"] == (r["gold"] == "A") for r in rows)
    assert all(r["p_known"] == 0.75 for r in rows)
    res = summary["results"]
    for name in ("triviaqa", "popqa"):
        for arm, m in res[name]["arms"].items():
            arm_rows = [r for r in rows if r["set"] == name and r["arm"] == arm]
            assert m["n"] == len(arm_rows) == 4
            assert m["accuracy"] == pytest.approx(
                sum(r["correct"] for r in arm_rows) / 4
            )
    again = dsx.run(offline, cfg)
    assert repr(again["results"]) == repr(res)


def test_orders_cover_every_permutation_of_small_label_sets() -> None:
    perms = set(itertools.permutations(["x", "y", "z"]))
    assert set(ost._orders(["x", "y", "z"], 6, "q")) == perms
