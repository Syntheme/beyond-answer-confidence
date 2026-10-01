import math

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.data.intents import Example, IntentDataset
from beyond_answer_confidence.experiments import option_count as oc
from beyond_answer_confidence.experiments import stated_odds as so
from beyond_answer_confidence.tasks import chance as ch
from beyond_answer_confidence.tasks import options as op
from beyond_answer_confidence.tasks.schema import order_collisions
from beyond_answer_confidence.tasks.synthetic import make_scenario


def test_device_ideals_are_distributions() -> None:
    for question, ideal, _, reps in ch.DEVICES.values():
        assert question.endswith("?")
        assert sum(ideal.values()) == pytest.approx(1.0)
        assert reps >= 1
    assert ch.DEVICES["two_dice_sum"][1]["7"] == pytest.approx(6 / 36)


def test_device_items_orders_and_replicates() -> None:
    devices = ch.device_items()
    counts: dict[str, int] = {}
    for it, _ in devices:
        counts[it.info["device"]] = counts.get(it.info["device"], 0) + 1
    assert counts["die_words"] == math.factorial(6)
    assert counts["two_dice_sum"] == 200
    assert counts["coin_heads"] == math.factorial(4)
    item, reps = next((it, r) for it, r in devices if it.info["device"] == "card_suit")
    assert reps == 5
    assert ch.order_replicate(item, 2, reps) == 2
    tasks = ch.device_tasks(devices)
    assert len(tasks) == sum(r for _, r in devices)
    assert order_collisions(tasks, "m") == 0
    assert len({(t.unit, t.request.replicate) for t in tasks}) == len(tasks)


def test_reordered_chances_list_a_tie_first() -> None:
    moved = 0
    for i in range(80):
        sc = make_scenario(i)
        if sc.profile not in ch.RELIST_PROFILES:
            continue
        facts, first = ch.reordered_chances(sc)
        top = max(sc.p_star.values())
        assert sc.p_star[first] == top
        assert f"{first} {round(top * 100)} %" in facts[-1]
        moved += first != sc.options[0]
    assert moved > 0
    items = ch.relisted_items(80)
    assert {it.info["profile"] for it in items} <= set(ch.RELIST_PROFILES)
    assert all(set(it.nouls) == {"determined", "settled"} for it in items)


def test_tv_of_mean_known_answers() -> None:
    ideal = {"a": 0.5, "b": 0.5}
    assert so.tv_of_mean([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}], ideal) == 0.0
    assert so.tv_of_mean([{"a": 1.0, "b": 0.0}], ideal) == 0.5
    res = so.tv_of_mean_interval(
        [{"a": 1.0, "b": 0.0}] * 4, ideal, 10, np.random.default_rng(0), 0.1
    )
    assert res["tv"] == 0.5
    assert res["tail"] == 1.0
    assert res["ci"] == [0.5, 0.5]


def test_stated_odds_analysis_on_position_biased_rows() -> None:
    rows = []
    for it, reps in ch.device_items():
        order = list(it.options)
        dist = {
            o: 0.5 if j == 0 else 0.5 / (len(order) - 1) for j, o in enumerate(order)
        }
        assert reps >= 1
        rows.append(
            {**it.info, "unit": it.unit, "dist": dist, "top": order[0], "p_max": 0.5}
        )
    rows += [
        {**it.info, "top": it.info["first_listed"], "p_max": 0.6}
        for it in ch.relisted_items(60)
    ]
    res = so.analyse(rows, so.Config(bootstrap=20, averaging_draws=3))
    assert res["fair_die_words"]["tv"] == pytest.approx(0.0, abs=1e-12)
    assert res["listed_first_minus_shown_first"]["share_top_is_first_listed"] == 1.0
    words = res["devices"]["die_words"]
    assert words["mean_probability_by_position"][0] == pytest.approx(0.5)
    assert words["permutation_averaging_tv"]["1"] > 0.1
    assert words["top_label_counts"]["one"] == 120


def test_parse_corrected() -> None:
    assert op.parse_corrected("2") == 2
    assert op.parse_corrected(" c ") == 2
    assert op.parse_corrected("2 and 3") is None
    assert op.parse_corrected("4") is None
    assert op.parse_corrected(None) is None


def test_fixed_order_and_exam_items() -> None:
    it = op.fixed_order_item("u", "Q?", [" x ", "y", "All of the above"], 1, {"s": 1})
    assert it.options == ("A", "B", "C")
    assert it.gold == "B"
    assert it.descriptions == {"A": "x", "B": "y", "C": "All of the above"}
    assert it.info["letter_of_index"] == {"0": "A", "1": "B", "2": "C"}
    redux = pd.DataFrame(
        [
            {
                "question": "Q",
                "choices": np.array(["a", "b", "c", "d"]),
                "answer": 3,
                "error_type": "wrong_groundtruth",
                "correct_answer": "A",
                "subject": "virology",
            }
        ]
    )
    (r,) = op.mmlu_redux_items(redux)
    assert r.unit == "redux:virology:0"
    assert r.gold == "D"
    assert r.info["corrected"] == 0
    cf = pd.DataFrame(
        [
            {
                "Question": "Q",
                "A": 1,
                "B": 2,
                "C": 3,
                "D": 4,
                "Answer": " C",
                "subject": "Math",
            }
        ]
    )
    (c,) = op.mmlu_cf_items(cf)
    assert c.gold == "C"
    assert c.descriptions == {"A": "1", "B": "2", "C": "3", "D": "4"}
    anli = pd.DataFrame(
        [{"uid": "x", "premise": "p", "hypothesis": "h", "label": 2, "round": 3}]
    )
    (a,) = op.anli_items(anli)
    assert a.gold == "contradiction"
    assert a.info == {"set": "anli", "round": 3}


def test_option_count_items() -> None:
    labels = tuple(f"l{i}" for i in range(8))
    test = tuple(Example(f"{lab} text {j}", lab) for lab in labels for j in range(3))
    ds = IntentDataset("d", labels, (), (), test)
    items = op.option_count_items(ds, 2, (3, 8))
    assert len(items) == 8 * 2 * 2
    for it in items:
        assert len(it.options) == it.info["K"] == len(set(it.options))
        assert it.gold in it.options
        assert it.state == {"customer_message": test[it.info["item"]].text}
    assert items == op.option_count_items(ds, 2, (3, 8))


def _option_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []

    def row(**kw: object) -> dict[str, object]:
        p = float(rng.uniform(0.3, 1))
        return {"p_max": p, "correct": bool(rng.random() < p), "p_known": p, **kw}

    for i in range(60):
        et = (
            "ok" if i < 40 else ("wrong_groundtruth" if i < 50 else "no_correct_answer")
        )
        rows.append(
            row(
                set="mmlu_redux",
                subject=f"s{i % 3}",
                error_type=et,
                corrected=1.0 if et == "wrong_groundtruth" else None,
                letter_of_index={"0": "A", "1": "B"},
                top="B" if i % 2 else "A",
            )
        )
        rows.append(row(set="mmlu_cf", subject="Math"))
        rows.append({**row(set="anli", round=1 + i % 3), "p_known": None})
        for k in (5, 10):
            rows.append(row(set="clinc_k", K=k, norm_entropy=0.5))  # noqa: PERF401
    return rows


def test_option_count_analysis() -> None:
    res = oc.analyse(_option_rows(), oc.Config(bootstrap=20))
    assert res["mmlu_redux_calibration"]["n"] == 40
    assert "error_auroc_p_known" in res["mmlu_redux_calibration"]
    assert "error_auroc_p_known" not in res["anli_calibration"]
    assert set(res["calibration_by_option_count"]["counts"]) == {5, 10}
    assert res["wrong_groundtruth_parsed"]["share_choosing_corrected"] == 0.5
    assert set(res["mmlu_redux_flawed"]) == {"wrong_groundtruth", "no_correct_answer"}
    assert res["clinc_by_K"][5]["mean_norm_entropy"] == 0.5
