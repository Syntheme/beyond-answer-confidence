from pathlib import Path

import numpy as np
import pytest

from beyond_answer_confidence.experiments import follow_up_questions as fq
from beyond_answer_confidence.tasks import follow_up as fu
from beyond_answer_confidence.tasks.schema import Item, item_tasks, order_collisions
from beyond_answer_confidence.tasks.synthetic import SETTLED_QUESTION, make_scenario


def test_adversarial_facts_and_percent_reading() -> None:
    sc = make_scenario(0)
    for cell in fu.ADVERSARIAL_CELLS:
        facts = fu.adversarial_facts(sc, cell)
        assert facts[: len(sc.fillers)] == list(sc.fillers)
        assert len(facts) == len(sc.fillers) + 1
    assert sc.options[0] in fu.adversarial_facts(sc, "ADVweather")[-1]
    assert sc.options[1] in fu.adversarial_facts(sc, "ADVweather")[-1]
    with pytest.raises(ValueError, match="D0"):
        fu.adversarial_facts(sc, "D0")
    assert fu.percent_reading(0) == pytest.approx(0.05)
    assert fu.percent_reading(9) == pytest.approx(0.95)
    assert fu.percent_reading(4.5) == pytest.approx(0.5)
    assert len(fu.PERCENT_LEVELS) == 10
    assert fu.PERCENT_LEVELS[0].startswith("0")


def test_wording_families_keep_the_original_question_first() -> None:
    assert fu.SETTLED_WORDINGS["settled"] == (SETTLED_QUESTION, False)
    assert fu.question_texts(fu.SETTLED_WORDINGS)["settled"] == SETTLED_QUESTION
    for fam in (fu.SETTLED_WORDINGS, fu.KNOWN_WORDINGS, fu.ENOUGH_WORDINGS):
        assert sum(rev for _, rev in fam.values()) == 1
    assert set(fu.SETTLED_POSITIVE_CELLS).isdisjoint(fu.SETTLED_NEGATIVE_CELLS)


def test_odds_format_items_formats_and_no_collisions() -> None:
    items = fu.odds_format_items(12)
    assert len(items) == 3 * 12
    assert {it.info["format"] for it in items} == set(fu.ODDS_FORMATS)
    for it in items:
        for _, levels in it.scores.values():
            assert len(levels) <= 10
    assert order_collisions(item_tasks(items, 3), "m") == 0
    noul = items[0]
    assert set(noul.nouls) == {"opt0", "opt1", "opt2", "opt3"}
    assert items[2].instructions == fu.INSTRUCTED_ODDS


def test_odds_readings_per_format() -> None:
    sc = make_scenario(1)
    noul = {"format": "noul", **{f"p_opt{k}": 0.1 * k for k in range(4)}}
    score = {"format": "score", **{f"s_opt{k}": float(k) for k in range(4)}}
    inst = {"format": "instructed", "dist": dict.fromkeys(sc.options, 0.25)}
    assert fu.odds_readings(noul, sc) == [0.0, 0.1, 0.2, pytest.approx(0.3)]
    assert fu.odds_readings(score, sc) == [0.05, 0.15, 0.25, 0.35]
    assert fu.odds_readings(inst, sc) == [0.25] * 4


def _entity_items() -> list[Item]:
    real = [
        Item(
            unit=f"popqa:{i}",
            state={"question": f"q{i}"},
            info={"set": "popqa", "s_pop": i, "fabricated": False},
        )
        for i in range(25)
    ]
    fab = [
        Item(
            unit=f"fab:rel:{j}",
            state={"question": f"f{j}"},
            info={"set": "fabricated", "fabricated": True},
        )
        for j in range(4)
    ]
    return real + fab


def _hotpot_items() -> list[Item]:
    return [
        Item(
            unit=f"hotpot:{c}:{i}",
            state={"question": "q", "paragraphs": c},
            options=("A", "B"),
            gold="A",
            info={"cell": c, "qid": str(i)},
        )
        for i in range(5)
        for c in ("closed", "dose0", "dose1", "dose2")
    ]


def test_popularity_sample_covers_every_quintile() -> None:
    picked = fu.popularity_sample(_entity_items(), 2)
    assert sorted(q for _, q in picked) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for it, q in picked:
        assert it.info["s_pop"] // 5 == q
    assert picked == fu.popularity_sample(_entity_items(), 2)


def test_wording_items_parts_and_units() -> None:
    items = fu.wording_items(
        _entity_items(),
        _hotpot_items(),
        n_scenarios=2,
        per_quintile=1,
        fabricated_per_relation=2,
        paragraph_questions=2,
    )
    parts = [it.info["part"] for it in items]
    assert parts.count("settled") == 2 * 9
    assert parts.count("known") == 5 + 2
    assert parts.count("enough") == 2 * 4
    adv = next(it for it in items if it.unit == "wording:settled:0:ADVrec")
    assert adv.info["positive"] is True
    assert set(adv.nouls) == set(fu.SETTLED_WORDINGS)
    known = [it for it in items if it.info["part"] == "known"]
    assert sum(it.info["fabricated"] for it in known) == 2
    assert all(it.unit.startswith("wording:") for it in items)


def test_cutoff_and_warned_items_from_stub_builders() -> None:
    news = [
        Item(
            unit=f"oracle_tf:{i}",
            state={"question": f"Did event {i} happen?"},
            options=("yes", "no"),
            gold="yes",
            nouls={"known": "k"},
            info={"set": "oracle_tf", "month": "2025-01" if i < 3 else "2023-01"},
        )
        for i in range(8)
    ]
    items = fu.cutoff_items(news)
    assert len(items) == 4 * 2 * 3  # 3 post + 3 sampled pre, four conditions each
    assert {it.info["cond"] for it in items} == set(fu.CUTOFF_CONDITIONS)
    base = next(it for it in items if it.info["cond"] == "base")
    assert base.request(0) == news[0].request(0)  # base repeats the source request
    dated = next(it for it in items if it.info["cond"] == "date")
    assert dated.state["today"] == fu.TODAY
    unk = next(it for it in items if it.info["cond"] == "date_unknown")
    assert unk.options == ("yes", "no", "unknown")
    assert unk.descriptions == fu.UNKNOWN_DESCRIPTIONS
    assert sum(it.info["post"] for it in items) == 12
    warned = fu.warned_paragraph_items(_hotpot_items())
    assert len(warned) == 15
    assert all(it.instructions == fu.PARAGRAPH_WARNING for it in warned)
    assert all(it.unit.startswith("warned:") for it in warned)


def test_second_look_item_shows_the_first_answer() -> None:
    it = Item(
        unit="u",
        state={"question": "q"},
        options=("A", "B"),
        descriptions={"A": "alpha", "B": "beta"},
    )
    look = fu.second_look_item(it, {"top": "B", "p_max": 0.7}, "u2", {"set": "s"})
    assert look.state == {"question": "q", "proposed_answer": "beta"}
    assert look.info == {
        "set": "s",
        "first_p_max": 0.7,
        "first_correct": False,
        "proposed": "B",
    }
    q = look.request(0).questions
    assert list(q) == ["correct", "chance"]
    assert q["chance"]["criteria"] == list(fu.PERCENT_LEVELS)
    bare = Item(unit="v", state={"s": 1})
    assert (
        fu.second_look_item(
            bare, {"top": "x", "p_max": 1, "correct": True}, "v", {}
        ).state["proposed_answer"]
        == "x"
    )


# --- analyses --------------------------------------------------------------------


def _wording_rows(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(30):
        for cell in (*fu.SETTLED_POSITIVE_CELLS, *fu.SETTLED_NEGATIVE_CELLS):
            pos = cell in fu.SETTLED_POSITIVE_CELLS
            p = rng.uniform(0.7, 1) if pos else rng.uniform(0, 0.3)
            row: dict[str, object] = {
                "part": "settled",
                "scenario": i,
                "cell": cell,
                "positive": pos,
            }
            for w, (_, rev) in fu.SETTLED_WORDINGS.items():
                row[f"p_{w}"] = 1 - p if rev else p
            rows.append(row)
    for j in range(80):
        fab = j % 4 == 0
        p = rng.uniform(0, 0.3) if fab else rng.uniform(0.5, 1)
        row = {"part": "known", "fabricated": fab, "qid": f"q{j}"}
        for w, (_, rev) in fu.KNOWN_WORDINGS.items():
            row[f"p_{w}"] = 1 - p if rev else p
        rows.append(row)
    for j in range(40):
        for cell in ("closed", "dose0", "dose1", "dose2"):
            p = rng.uniform(0.7, 1) if cell == "dose2" else rng.uniform(0, 0.4)
            row = {"part": "enough", "qid": f"h{j}", "cell": cell}
            for w, (_, rev) in fu.ENOUGH_WORDINGS.items():
                row[f"p_{w}"] = 1 - p if rev else p
            rows.append(row)
    return rows


def test_analyse_wording_on_clean_signals() -> None:
    rows = _wording_rows(np.random.default_rng(0))
    res = fq.analyse_wording(rows, fq.Config(bootstrap=50))
    for fam in ("settled_wordings", "known_wordings", "enough_wordings"):
        assert res[fam]["tail"] == 0.0
        assert all(v["auroc"] == 1.0 for v in res[fam]["wordings"].values())
        assert res[fam]["p_holm"] == 0.0
    assert res["settled_adversarial_cells"]["auroc"] == 1.0
    assert set(res["known_by_group"]) == {"made_up", "real"}
    assert "supported" not in str(res)


def test_analyse_odds_perfect_yes_no_and_collapsed_choice() -> None:
    rows = []
    for i in range(40):
        sc = make_scenario(i)
        star = [sc.p_star[o] for o in sc.options]
        base = {"scenario": i}
        rows.append(
            {**base, "format": "noul", **{f"p_opt{k}": s for k, s in enumerate(star)}}
        )
        rows.append(
            {
                **base,
                "format": "score",
                **{f"s_opt{k}": s * 10 - 0.5 for k, s in enumerate(star)},
            }
        )
        top = sc.options[int(np.argmax(star))]
        rows.append(
            {
                **base,
                "format": "instructed",
                "dist": {o: float(o == top) for o in sc.options},
            }
        )
    res = fq.analyse_odds(rows, fq.Config(bootstrap=50))
    assert res["per_option_yes_no"]["mean"] == pytest.approx(0.0)
    assert res["per_option_score"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert res["per_option_yes_no"]["tail"] == 0.0
    assert res["instructed_choice"]["mean"] > 0.1
    assert res["by_format"]["noul"]["mean_sum_of_readings"] == pytest.approx(1.0)
    assert res["by_format"]["noul"]["corr_with_p_star"] == pytest.approx(1.0)


def _cutoff_rows(rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in range(200):
        post = k % 2 == 0
        gold = "yes" if k % 3 else "no"
        common = {"post": post, "gold": gold, "p_known": 0.2}
        for cond in ("base", "date"):
            gap = 0.3 if cond == "base" else 0.2
            correct = not post
            p = min(1.0, 0.7 + (gap if post else 0.0))
            rows.append(
                {
                    **common,
                    "unit": f"cutoff:{cond}:oracle_tf:{k}",
                    "cond": cond,
                    "p_max": p,
                    "correct": correct,
                    "top": gold if correct else ("no" if gold == "yes" else "yes"),
                    "dist": {"yes": 0.5, "no": 0.5},
                }
            )
        for cond in ("unknown", "date_unknown"):
            p_unk = rng.uniform(0.6, 1) if post else rng.uniform(0, 0.4)
            rows.append(
                {
                    **common,
                    "unit": f"cutoff:{cond}:oracle_tf:{k}",
                    "cond": cond,
                    "p_max": max(p_unk, 1 - p_unk),
                    "correct": False,
                    "top": "unknown" if post else gold,
                    "dist": {
                        "yes": (1 - p_unk) / 2,
                        "no": (1 - p_unk) / 2,
                        "unknown": p_unk,
                    },
                }
            )
    return rows


def test_analyse_cutoff_on_synthetic_rows() -> None:
    res = fq.analyse_cutoff(
        _cutoff_rows(np.random.default_rng(2)), fq.Config(bootstrap=50)
    )
    assert res["date_changes_post_cutoff_gap"]["mean"] == pytest.approx(-0.1)
    assert res["unknown_option_flags_post_cutoff"]["auroc"] == 1.0
    assert res["confident_errors_with_unknown_option"]["mean"] == 0.0
    by = res["by_condition"]
    assert by["unknown:post"]["share_unknown"] == 1.0
    assert by["unknown:post"]["accuracy_answered"] is None
    assert by["date_unknown_auroc_p_unknown"] == 1.0


def test_analyse_warned_compares_with_unwarned_rows() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for j in range(300):
        for cell in ("dose0", "dose1"):
            p = float(rng.uniform(0.5, 1))
            rows.append(
                {
                    "cell": cell,
                    "p_max": p,
                    "correct": bool(rng.random() < p),
                    "p_enough": 0.3,
                    "qid": f"q{j}",
                }
            )
    unwarned = [
        {"set": "hotpot", "cell": c, "correct": True, "p_max": 0.9}
        for c in ("closed", "dose0", "dose1", "dose2")
    ]
    res = fq.analyse_warned(rows, fq.Config(bootstrap=30), unwarned)
    cells = res["calibration_with_warning"]["cells"]
    assert set(cells) == {"dose0", "dose1"}
    assert all(c["smece"] < 0.1 for c in cells.values())
    assert res["by_cell"]["closed"]["warned_accuracy"] is None
    assert res["by_cell"]["dose0"]["unwarned_accuracy"] == 1.0
    assert "by_cell" not in fq.analyse_warned(rows, fq.Config(bootstrap=5))


def test_probe_of_and_mean_by() -> None:
    assert fq.probe_of("wording:known:popqa:1") == "wording"
    assert fq.probe_of("odds:noul:3") == "odds_formats"
    assert fq.probe_of("cutoff:base:oracle_tf:9") == "cutoff"
    assert fq.probe_of("warned:hotpot:dose0:1") == "warned_paragraphs"
    rows = [{"g": "b", "v": 1.0}, {"g": "a", "v": 2.0}, {"g": "b", "v": 3.0}]
    assert fq.mean_by(rows, "g", lambda r: r["v"]) == {"a": 2.0, "b": 2.0}
    assert list(fq.mean_by(rows, "g", lambda r: r["v"])) == ["a", "b"]


def test_unwarned_rows_missing_file(tmp_path: Path) -> None:
    from beyond_answer_confidence.settings import Settings

    assert fq._unwarned_rows(Settings(output_dir=tmp_path)) is None
