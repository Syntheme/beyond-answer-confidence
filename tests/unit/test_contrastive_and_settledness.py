import numpy as np
import pytest

from beyond_answer_confidence.experiments import contrastive_facts as cf
from beyond_answer_confidence.experiments import inferred_settledness as ins
from beyond_answer_confidence.tasks import contrastive as ct
from beyond_answer_confidence.tasks import settledness as st
from beyond_answer_confidence.tasks.schema import item_tasks, order_collisions


def test_word_diff_known_answers() -> None:
    assert ct.word_diff("a b c", "a b c") == 0
    assert ct.word_diff("a b c", "a x c") == 1
    assert ct.word_diff("a b", "a b c d") == 2
    assert ct.word_diff("a b c d", "a x y d") == 2


def test_fact_sentence_numeric_ranges() -> None:
    import random

    fact = ct.POLICIES[0].facts[0]
    for seed in range(20):
        rng = random.Random(seed)  # noqa: S311
        days = int(ct.fact_sentence(fact, True, "Ben", rng).split()[4])
        assert 2 <= days <= 29
        days = int(ct.fact_sentence(fact, False, "Ben", rng).split()[4])
        assert 31 <= days <= 90
    assert ct.fact_sentence(ct.POLICIES[2].facts[0], True, "Kemi", rng).startswith(
        "Kemi completed"
    )


@pytest.mark.parametrize("policy", ct.POLICIES, ids=lambda p: p.name)
def test_cases_are_well_defined(policy: ct.Policy) -> None:
    for i in range(15):
        case = ct.make_case(policy, i)
        assert list(case.variants) == list(ct.VARIANTS)
        label = policy.rule(*case.values)
        outcome = policy.outcomes[0] if label else policy.outcomes[1]
        base_facts, base_ideal = case.variants["base"]
        assert base_ideal == outcome
        cf_facts, cf_ideal = case.variants["counterfactual"]
        assert cf_ideal != outcome
        assert cf_ideal in policy.outcomes
        diff = [a != b for a, b in zip(base_facts, cf_facts, strict=True)]
        assert sum(diff) == 1
        assert case.variants["deleted"][1] == ct.CANNOT_TELL
        assert len(case.variants["deleted"][0]) == len(base_facts) - 1
        assert case.variants["contradictory"][0][:-1] == base_facts
        assert case.variants["irrelevant"][1] == outcome
        assert base_facts[0].startswith(case.name)
    assert ct.make_case(policy, 3) == ct.make_case(policy, 3)
    assert ct.make_case(policy, 3, seed=1) != ct.make_case(policy, 3)


def test_contrastive_items_shape_and_requests() -> None:
    cases, items = ct.contrastive_items(4)
    assert len(cases) == 6 * 4
    assert len(items) == 6 * 4 * 5
    first = items[0]
    assert first.unit == "refund:0:base"
    assert first.options == ("refund", "no_refund", ct.CANNOT_TELL)
    assert set(first.nouls) == {"outcome", "enough"}
    assert first.info["rule_label"] in first.options
    assert order_collisions(item_tasks(items, 3), "m") == 0


def _contrastive_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases, items = ct.contrastive_items(10)
    del cases
    for it in items:
        var = it.info["variant"]
        gold = it.gold
        dist = dict.fromkeys(it.options, 0.0)
        dist[str(gold)] = 1.0
        rows.append(
            {
                **it.info,
                "unit": it.unit,
                "gold": gold,
                "dist": dist,
                "top": gold,
                "p_gold": 1.0,
                "p_outcome": 0.95,
                "p_enough": 0.1 if var in ("deleted", "contradictory") else 0.9,
            }
        )
    return rows


def test_contrastive_analysis_on_ideal_answers() -> None:
    res = cf.analyse(_contrastive_rows(), cf.Config(bootstrap=40))
    assert res["follows_evidence"]["variants"]["base"]["mean"] == 1.0
    assert res["follows_evidence"]["tail"] == 0.0
    assert res["deleted_gives_cannot_tell"]["mean"] == 1.0
    assert res["irrelevant_fact_shift"]["mean"] == 0.0
    assert res["enough_flags_undecidable"]["auroc"] == 1.0
    assert res["enough_auroc_deleted_only"] == 1.0
    assert res["deleted_share_top_not_cannot_tell"] == 0.0
    assert res["by_variant"]["base"]["share_outcome_noul_extreme"] == 1.0
    assert set(res["by_policy"]) == {p.name for p in ct.POLICIES}


def test_settled_items_differ_only_in_the_cue() -> None:
    items = st.settledness_items()
    assert len(items) == len(st.EVENT_KINDS) * 60 * 4 * 2
    by_unit = {it.unit: it for it in items}
    past, future = by_unit["tense:court:3:past"], by_unit["tense:court:3:future"]
    assert past.options == future.options
    assert past.state["question"] == future.state["question"]
    assert past.state["facts"][1] == future.state["facts"][1]
    assert "was heard" in past.state["facts"][0]
    assert "will be heard" in future.state["facts"][0]
    assert by_unit["date_today:court:3:past"].state["facts"][0] == st.TODAY
    assert st.TODAY not in by_unit["date_only:court:3:past"].state["facts"]
    for it in items:
        text = " ".join(it.state["facts"]).lower()
        if it.info["cond"] != "explicit":
            assert "decided" not in text
    ev = st.make_event(st.EVENT_KINDS[0], 0)
    assert len(set(ev.contenders)) == 4
    assert 1985 <= ev.past_year <= 2015 < 2031 <= ev.future_year <= 2045
    assert ev.phrase == f"the {ev.name} Cup final"


def test_settled_analysis_on_separable_answers() -> None:
    items = [
        it
        for it in st.settledness_items(10)
        if it.info["domain"] in ("sports", "lottery")
    ]
    rng = np.random.default_rng(0)
    fake = [
        {
            **it.info,
            "unit": it.unit,
            "p_settled": float(
                rng.uniform(0.6, 1)
                if it.info["status"] == "past"
                else rng.uniform(0, 0.4)
            ),
            "p_max": 0.4,
            "top": it.info["first_listed"],
        }
        for it in items
    ]
    res = ins.analyse(fake, ins.Config(bootstrap=50))
    for name in ("tense", "date_with_today", "lottery_tense_and_date"):
        assert res[name]["auroc"] == 1.0
        assert res[name]["p"] == 0.0
        assert "p_holm" in res[name]
    assert "p_holm" not in res["explicit"]
    assert res["choice"]["tense"]["share_first_listed"] == 1.0
    assert res["choice"]["tense"]["share_p_max_ge_0.5"] == 0.0
    assert res["mean_p_settled"]["tense"]["past"] > 0.6
