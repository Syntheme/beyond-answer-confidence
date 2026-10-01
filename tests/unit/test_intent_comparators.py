import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from intent_helpers import LABELS, dist, row, tiny_dataset, without_oos

from beyond_answer_confidence.analyses import error_detection as ed
from beyond_answer_confidence.analyses.knowledge_dial import aggregate
from beyond_answer_confidence.experiments.comparators import deep_ensemble as de
from beyond_answer_confidence.experiments.comparators import gliner
from beyond_answer_confidence.stats.resampling import bootstrap, on_resample
from beyond_answer_confidence.tasks import out_of_scope as oos_tasks
from beyond_answer_confidence.tasks.intents import sample_examples

# --- deep ensemble ---------------------------------------------------------------


def test_decompose_identical_members_have_zero_mi() -> None:
    d = de.decompose(np.array([[[0.7, 0.2, 0.1]]] * 5))
    assert d["mi"][0] == pytest.approx(0.0, abs=1e-9)
    assert d["total"][0] == pytest.approx(d["aleatoric"][0])


def test_decompose_disagreeing_confident_members_are_epistemic() -> None:
    d = de.decompose(np.array([[[1.0, 0.0]], [[0.0, 1.0]]]))
    assert d["aleatoric"][0] == pytest.approx(0.0, abs=1e-6)
    assert d["mi"][0] == pytest.approx(np.log(2), abs=1e-6)


def test_epochs_for() -> None:
    assert de.epochs_for(10_000, full=True) == de.FULL_EPOCHS
    assert de.epochs_for(77, full=False) == 134  # 3 steps per epoch, >= 400 steps
    assert de.epochs_for(10, full=False) == de.MAX_EPOCHS
    assert de.epochs_for(100_000, full=False) == de.FULL_EPOCHS


def test_setups_without_oos_split() -> None:
    ds = without_oos(tiny_dataset(n_labels=20, name="banking77"))
    setups = {s.name: s for s in de.setups_for(ds)}
    assert set(setups) == {"full", "heldout", "k1", "k2", "k4", "k8"}
    assert len(setups["heldout"].labels) == 10
    assert all(ex.label in setups["heldout"].labels for ex in setups["heldout"].train)
    assert len(setups["k2"].train) == 2 * 20
    assert setups["k1"].eval_ids == setups["full"].eval_ids


def test_setups_with_oos_split_evaluate_oos() -> None:
    ds = tiny_dataset(n_labels=6, name="clinc150")
    setups = {s.name: s for s in de.setups_for(ds)}
    assert "heldout" not in setups
    assert sum(i.startswith("oos:") for i in setups["full"].eval_ids) == len(
        ds.oos_test
    )
    assert not any(i.startswith("oos:") for i in setups["k8"].eval_ids)


def test_scope_id() -> None:
    assert de.scope_id("clinc150:t:in:12") == "test:12"
    assert de.scope_id("clinc150:oos:3") == "oos:3"


def _ensemble_file(tmp_path: Path) -> Path:
    confident = [0.9, 0.05, 0.05]
    probs = np.array(
        [[confident, [0.05, 0.9, 0.05], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
         [confident, [0.05, 0.9, 0.05], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
    )  # fmt: skip
    path = tmp_path / "e.npz"
    np.savez(
        path,
        probs=probs,
        labels=np.array(["a", "b", "c"]),
        eval_ids=np.array(["test:0", "test:1", "oos:0", "oos:1"]),
    )
    return path


def test_ensemble_frame_maps_seed_form_ids_to_item_keys(tmp_path: Path) -> None:
    keys = [oos_tasks.item_key("x", k == "oos", i) for k, i in
            (("in", 0), ("in", 1), ("oos", 0), ("oos", 1))]  # fmt: skip
    path = tmp_path / "held.npz"
    np.savez(
        path,
        probs=np.full((2, 4, 3), 1 / 3),
        labels=np.array(["a", "b", "c"]),
        eval_ids=np.array([oos_tasks.seed_key(k) for k in keys]),
    )
    ens = de.ensemble_frame(path, dict.fromkeys(keys, "a"))
    assert ens["id"].tolist() == keys


def test_ensemble_frame_and_oos_comparison(tmp_path: Path) -> None:
    gold = {"test:0": "a", "test:1": "b", "oos:0": None, "oos:1": None}
    ens = de.ensemble_frame(_ensemble_file(tmp_path), gold)
    assert ens["correct"].tolist() == [True, True, False, False]
    assert (ens.loc[ens["id"].str.startswith("oos"), "mi"] > 0.5).all()
    items = pd.DataFrame(
        {
            "key": ["x:in:0", "x:in:1", "x:oos:0", "x:oos:1"],
            "oos": [False, False, True, True],
            "gold": ["a", "b", None, None],
            "closed_p_max": [0.9, 0.9, 0.5, 0.5],
            "p_oos": [0.0, 0.0, 0.9, 0.9],
            "p_fits": [1.0, 1.0, 0.1, 0.1],
        }
    )
    id_map = {k: de.scope_id(k) for k in items["key"]}
    res = de.oos_comparison(ens, items, id_map, b=20, rng=np.random.default_rng(0))
    assert res["ens_mi"]["auroc"] == 1.0
    assert res["jev_with_oos_p_oos"]["auroc"] == 1.0
    assert res["paired_best_jev_minus_best_ens"]["diff"] == pytest.approx(0.0)
    only = de.ensemble_oos_only(ens, b=20, rng=np.random.default_rng(0))
    assert only["ens_mi"]["auroc"] == 1.0
    assert only["n_oos"] == 2


def test_on_resample_pairs_arrays() -> None:
    x, y = np.arange(5.0), np.arange(5.0) * 2
    stat = on_resample(lambda a, b: float((b - 2 * a).sum()), x, y)
    assert stat(np.array([0, 4, 4])) == 0.0
    vals = bootstrap(5, stat, 10, np.random.default_rng(0))
    assert vals.tolist() == [0.0] * 10


# --- GLiNER ----------------------------------------------------------------------


def test_sanitise_and_describe() -> None:
    assert gliner.sanitise("card (new) [L] please") == "card new please"
    assert gliner.describe(None) is None
    assert gliner.describe({"name": "x_y"}) == "name: x_y"
    assert gliner.describe({"examples": ["a (b)", "c"]}) == "examples: a b | c"


def test_labels_use_the_knowledge_dial_codes() -> None:
    ds = tiny_dataset(n_labels=5, name="banking77")
    ex = sample_examples(ds, 0)
    l0, m0 = gliner.labels_for(ds, 0, "L0", ex)
    assert isinstance(l0, list)
    assert sorted(m0.values()) == sorted(ds.labels)
    ln, mn = gliner.labels_for(ds, 0, "Lname", ex)
    assert isinstance(ln, dict)
    assert mn == m0
    assert all(ln[c] == f"name: {m0[c]}" for c in ln)
    names, mapping = gliner.labels_for(ds, 0, gliner.NAMES, ex)
    assert names == list(ds.labels)
    assert mapping == {lab: lab for lab in ds.labels}


def test_row_format_resume_and_items(tmp_path: Path) -> None:
    r = gliner.row_for(3, "L0", "a", {"a": 0.2, "b": 0.8}, "b", 400)
    assert r["choice"] == "b"
    assert r["p_gold"] == pytest.approx(0.2)
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(r) + "\n")
    assert gliner.done_keys(path) == {(3, "L0")}
    assert gliner.done_keys(tmp_path / "missing.jsonl") == set()
    ds = tiny_dataset(n_labels=4)
    cfg = gliner.InferenceConfig(per_intent=2, subset_per_intent=1)
    assert len(gliner.choose_items(ds, cfg)) == 4
    assert len(gliner.choose_items(ds, gliner.InferenceConfig(limit=3))) == 3


def gliner_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    gl_rows, api_rows = [], []
    for item in range(60):
        gold = LABELS[1 + item % (len(LABELS) - 1)]
        gl_rows.append(row(item, "L0", dist(gold, 1 / len(LABELS)), gold, arm="gliner"))
        gl_rows.append(row(item, "Lname", dist(gold, 0.9), gold, arm="gliner"))
        gl_rows.append(row(item, "L1", dist(gold, 0.8), gold, arm="gliner"))
        gl_rows.append(row(item, "names", dist(gold, 0.9), gold, arm="gliner"))
        for cond, p in (("L0", None), ("Lname", 0.9), ("L1", 0.8)):
            probs = dist(gold, 0.02, LABELS[0], 0.4) if p is None else dist(gold, p)
            api_rows.append(row(item, cond, probs, gold))
    for r in gl_rows:
        r["n_tokens"] = 500
    return pd.DataFrame(gl_rows), pd.DataFrame(api_rows)


def test_compare_rows() -> None:
    gl, api = gliner_rows()
    res = gliner.compare_rows(gl, api, b=100)
    assert res["gliner_no_knowledge"]["mean_p_max"] == pytest.approx(1 / len(LABELS))
    assert res["l0_p_max_jev_minus_gliner"]["mean"] > 0.3
    assert res["gliner_entropy_minus_l0"]["Lname"]["mean"] < 0
    assert res["gliner_names"]["n"] == 60
    assert res["n_tokens_median"] == dict.fromkeys(("L0", "L1", "Lname", "names"), 500)
    assert set(res["smece"]["gliner"]) == {"Lname", "L1"}
    assert res["reliability_bins"]["names"]["jev"] == []
    json.dumps(res)


# --- error detection -------------------------------------------------------------


def test_replicate_features_flags_disagreement() -> None:
    rows = pd.DataFrame(
        [
            row(0, "L1", dist("i1", 0.9), "i1", 0),
            row(0, "L1", dist("i1", 0.1, "i2", 0.8), "i1", 1),
        ]
    )
    f = ed.replicate_features(rows)
    assert f["rep_flip"].iloc[0]
    assert f["rep_tvd"].iloc[0] > 0.5


def test_error_detection_and_replicate_signal() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for item in range(80):
        gold = LABELS[1 + item % 19]
        other = LABELS[(LABELS.index(gold) + 1) % 20]
        for rep in range(2):
            ok = rng.uniform() < 0.7
            p = float(rng.uniform(0.4, 0.95))
            probs = dist(gold, p) if ok else dist(gold, 0.0, other, p)
            rows.append(row(item, "L1", probs, gold, rep))
    frame = pd.DataFrame(rows)
    cells = aggregate(frame)
    det = ed.error_detection(cells, 50, np.random.default_rng(0))["L1"]
    assert det["n"] == 80
    assert 0 <= det["auroc_error"] <= 1
    assert det["auroc_error_ci95"][0] <= det["auroc_error_ci95"][1]
    sig = ed.replicate_signal(cells, ed.replicate_features(frame), 20, rng)
    assert "gain" in sig["L1"]
    assert sig["L2"]["skipped"].startswith("fewer than")
    bins = ed.knowledge_bins(cells)
    assert sum(b["n"] for b in bins["L1"]) == 80


def test_error_structure_counts_pairs(tmp_path: Path) -> None:
    gold_of = dict.fromkeys(range(4), LABELS[1])
    wrong = dist(LABELS[1], 0.1, LABELS[2], 0.8)
    rows = [
        row(item, cond, wrong, LABELS[1])
        for item in range(4)
        for cond in ("Lname", "L8", "Lname+8")
    ]
    probs = np.zeros((2, 4, len(LABELS)))
    probs[:, :, 2] = 0.7
    probs[:, :, 1] = 0.3
    ens = tmp_path / "e.npz"
    np.savez(
        ens,
        probs=probs,
        labels=np.array(LABELS),
        eval_ids=np.array([f"test:{i}" for i in range(4)]),
    )
    res = ed.error_structure(pd.DataFrame(rows), ens, gold_of, len(LABELS))["L8"]
    assert res["jev_errors"] == 4
    assert res["errors_gold_is_second_choice"] == 1.0
    assert res["errors_in_ensemble_top2"] == 1.0
    assert res["top_confused_pairs"][0] == {
        "gold": LABELS[1], "chosen": LABELS[2], "jev": 4, "ensemble": 4
    }  # fmt: skip
    json.dumps(res)
