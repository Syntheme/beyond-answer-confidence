from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.data import multiple_choice as mcd
from beyond_answer_confidence.experiments import open_benchmarks as ob
from beyond_answer_confidence.experiments import second_look as sl
from beyond_answer_confidence.stats.inference import mean_interval, mean_interval_lower
from beyond_answer_confidence.tasks.schema import Item


def _mc(unit: str, n: int = 4) -> Item:
    letters = tuple("ABCD"[:n])
    return Item(
        unit=unit,
        state={"question": f"{unit}?"},
        options=letters,
        gold="A",
        descriptions={x: f"{unit}-{x}" for x in letters},
        info={"set": "src"},
    )


def test_simpleqa_idk_items_add_a_fifth_option() -> None:
    (it,) = ob.simpleqa_idk_items([_mc("simpleqa:1")])
    assert it.unit == "idk:simpleqa:1"
    assert it.options == ("A", "B", "C", "D", "E")
    assert it.descriptions is not None
    assert it.descriptions["E"] == ob.IDK
    assert it.info["idk"] == "E"
    assert it.info["benchmark_unit"] == "simpleqa:1"
    with pytest.raises(ValueError, match="descriptions"):
        ob.simpleqa_idk_items([Item(unit="x", state={}, options=("A",))])


def test_verification_items_pair_gold_and_distractor() -> None:
    items = ob.verification_items([_mc(f"t{i}") for i in range(5)], n=3)
    assert len(items) == 6
    true, false = items[0], items[1]
    q = true.info["question_unit"]
    assert true.state == {"question": f"{q}?", "proposed_answer": f"{q}-A"}
    assert false.state["proposed_answer"] == f"{q}-B"
    assert (true.info["truth"], false.info["truth"]) == (True, False)
    assert list(true.nouls) == ["correct"]
    with pytest.raises(ValueError, match="gold"):
        ob.verification_items([Item(unit="x", state={}, options=("A",))], n=1)


def _ambig_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "amb",
                "question": "Which?",
                "annotations": {
                    "type": np.array(["multipleQAs"]),
                    "qaPairs": [{"answer": [np.array(["Oak"]), np.array(["Elm"])]}],
                },
            },
            {
                "id": "dup",
                "question": "Same?",
                "annotations": {
                    "type": np.array(["multipleQAs"]),
                    "qaPairs": [{"answer": [np.array(["Oak"]), np.array(["oak."])]}],
                },
            },
            {
                "id": "one",
                "question": "Single?",
                "annotations": {
                    "type": np.array(["singleAnswer"]),
                    "answer": [np.array(["  ", "Ash"])],
                },
            },
            {
                "id": "none",
                "question": "Empty?",
                "annotations": {
                    "type": np.array(["singleAnswer"]),
                    "answer": [np.array([""])],
                },
            },
        ]
    )


def test_ambiguity_candidates_and_items() -> None:
    cands = ob.ambiguity_candidates(_ambig_frame())
    assert [(c["id"], c["valid"], c["kind"]) for c in cands] == [
        ("amb", ["Oak", "Elm"], "ambiguous"),
        ("one", ["Ash"], "single"),
    ]

    def pick(
        golds: Sequence[str], excludes: Sequence[Sequence[str]], pool: Sequence[str]
    ) -> list[list[str] | None]:
        assert list(pool) == ["Ash", "Elm", "Oak"]
        return [["d1", "d2", "d3"], None][: len(golds)]

    (item,) = ob.ambiguity_items(cands, pick)
    assert item.unit == "ambig:amb"
    assert len(item.options) == 4
    assert item.descriptions is not None
    valid = {item.descriptions[k] for k in item.info["valid"]}
    assert valid == {"Oak", "Elm"}
    assert item.descriptions[str(item.gold)] == "Oak"


def test_valid_spread_known_answers() -> None:
    assert ob.valid_spread({"A": 0.4, "B": 0.4, "C": 0.2}, ["A", "B"]) == 1.0
    assert ob.valid_spread({"A": 1.0, "B": 0.0}, ["A", "B"]) == 0.0
    assert ob.valid_spread({"A": 0.0, "B": 0.0, "C": 1.0}, ["A", "B"]) == 0.0


def test_open_benchmark_analysis() -> None:
    rows: list[dict[str, object]] = []
    ref = []
    for i in range(40):
        wrong = i % 2 == 0
        ref.append({"unit": f"s{i}", "correct": not wrong})
        p_idk = 0.8 if wrong else 0.1
        rows.append(
            {
                "set": "simpleqa_idk",
                "idk": "E",
                "benchmark_unit": f"s{i}",
                "dist": {"A": 1 - p_idk, "E": p_idk},
                "top": "E" if wrong else "A",
                "correct": not wrong,
            }
        )
        for truth in (True, False):
            rows.append(  # noqa: PERF401
                {
                    "set": "trivia_verify",
                    "truth": truth,
                    "p_correct": 0.95 if truth else 0.05,
                }
            )
        kind = "ambiguous" if i % 2 else "single"
        rows.append(
            {
                "set": "ambig_spread",
                "kind": kind,
                "valid": ["A", "B"] if kind == "ambiguous" else ["A"],
                "dist": {"A": 0.5, "B": 0.5, "C": 0.0},
                "p_max": 0.5,
                "correct": True,
            }
        )
    res = ob.analyse(rows, ref, ob.Config(bootstrap=20))
    assert res["idk_flags_benchmark_errors"]["auroc"] == 1.0
    assert res["false_answers_confidently_accepted"]["mean"] == 0.0
    assert res["verification_calibration"]["accuracy"] == 1.0
    assert res["ambiguity_spreads_probability"]["mean"] == 1.0
    assert res["simpleqa_idk"]["abstention_rate"] == 0.5
    assert res["simpleqa_idk"]["accuracy_answered"] == 1.0
    assert res["simpleqa_idk"]["multiple_choice_accuracy"] == 0.5
    assert res["trivia_verify"]["auroc_true_vs_false"] == 1.0
    assert res["ambig_spread"]["mean_mass_on_valid"] == 1.0


def test_second_look_synthetic_items_and_ideals() -> None:
    from beyond_answer_confidence.tasks.synthetic import make_scenario

    rows = []
    for i in range(3):
        sc = make_scenario(i)
        for cell in ("D0", "D3", "SFp", "SP"):
            rows.append(  # noqa: PERF401
                {
                    "scenario": i,
                    "cell": cell,
                    "alone": False,
                    "top": sc.gold,
                    "p_max": 0.6,
                }
            )
        rows.append(
            {"scenario": i, "cell": "D0", "alone": True, "top": "x", "p_max": 1}
        )
    items = sl.synthetic_items(rows, 3)
    assert len(items) == 12
    by = {it.unit: it for it in items}
    assert by["synthetic:0:D0"].info["ideal"] == 0.25
    assert by["synthetic:0:D3"].info["ideal"] == 1.0
    assert by["synthetic:0:D3"].info["first_correct"] is True
    sc = make_scenario(0)
    assert by["synthetic:0:SFp"].info["ideal"] == sc.p_star[sc.gold]
    assert by["synthetic:0:D0"].state["proposed_answer"] == sc.gold


def test_source_rows_must_exist(tmp_path: Path) -> None:
    from beyond_answer_confidence.experiments.base import source_rows
    from beyond_answer_confidence.settings import Settings

    settings = Settings(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="benchmarks"):
        source_rows(settings, "benchmarks")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "rows.jsonl").write_text("")
    assert source_rows(settings, "benchmarks").name == "rows.jsonl"


def test_second_look_analysis() -> None:
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    for i in range(120):
        ok = bool(i % 3)
        base = {"first_correct": ok, "first_p_max": 0.9, "s_chance": 8.0 if ok else 1.0}
        pc = 0.9 if ok else 0.1
        rows.append(
            {
                **base,
                "set": "popqa",
                "made_up": i < 20,
                "quintile": i % 5,
                "p_correct": pc,
            }
        )
        rows.append({**base, "set": "oracle_tf", "post": i % 2 == 0, "p_correct": pc})
        rows.append(
            {
                **base,
                "set": "synthetic",
                "cell": ["D0", "D3", "SFp", "SP"][i % 4],
                "ideal": 0.25,
                "p_correct": 0.25 + float(rng.normal(0, 0.01)),
            }
        )
    res = sl.analyse(rows, sl.Config(bootstrap=30))
    gain = res["error_detection_gain"]["pools"]["popqa_plus_made_up"]
    assert gain["auroc_first_p_max"] == 0.5
    assert gain["auroc_p_correct"] == 1.0
    assert gain["gain"] == 0.5
    assert gain["tail"] == 0.0
    assert res["synthetic_ideal_error"]["cells"]["D0"]["mean"] < 0.02
    assert set(res["synthetic_ideal_error"]["cells"]) == {"D0", "SFp"}
    assert res["post_cutoff_gap"]["mean"] == pytest.approx(
        np.mean([(0.9 if i % 3 else 0.1) - bool(i % 3) for i in range(0, 120, 2)])
    )
    assert set(res["by_set"]) == {"popqa", "oracle_tf", "synthetic"}
    assert res["popqa_by_quintile"][0]["mean_first_p_max"] == 0.9
    assert set(res["oracle_by_period"]) == {"pre", "post"}


def test_mean_interval_lower_mirrors_mean_interval() -> None:
    v = np.array([0.2, 0.4, 0.6, 0.8])
    lo = mean_interval_lower(v, 200, np.random.default_rng(3), 0.5)
    hi = mean_interval(v, 200, np.random.default_rng(3), threshold=0.5)
    assert lo["mean"] == hi["mean"] == 0.5
    assert lo["ci"] == hi["ci"]
    # Every replicate is <= 0.5 or >= 0.5; exactly-equal ones count in both.
    assert lo["tail"] + hi["tail"] >= 1.0
    assert (
        mean_interval_lower(np.ones(5), 10, np.random.default_rng(0), 0.9)["tail"]
        == 0.0
    )


def test_multiple_choice_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    def fake_download(repo: str, filename: str, **kw: object) -> str:
        assert kw["revision"] == mcd.REVISIONS[repo]
        path = tmp_path / repo.replace("/", "_") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".arrow"):
            table = pa.table({"question": ["q"], "answer": [1]})
            with (
                pa.OSFile(str(path), "wb") as sink,
                pa.ipc.new_stream(sink, table.schema) as writer,
            ):
                writer.write_table(table)
        else:
            pd.DataFrame({"x": [1, 2]}).to_parquet(path)
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    redux = mcd.load_mmlu_redux(tmp_path)
    assert len(redux) == len(mcd.MMLU_REDUX_SUBJECTS)
    assert list(redux["subject"]) == list(mcd.MMLU_REDUX_SUBJECTS)
    cf = mcd.load_mmlu_cf(tmp_path)
    assert len(cf) == 2 * len(mcd.MMLU_CF_SUBJECTS)
    anli = mcd.load_anli(tmp_path)
    assert sorted(set(anli["round"])) == [1, 2, 3]
