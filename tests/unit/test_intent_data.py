import hashlib
from collections import Counter
from pathlib import Path

import pytest
from intent_helpers import tiny_dataset

from beyond_answer_confidence.data import intents
from beyond_answer_confidence.data.download import IntegrityError
from beyond_answer_confidence.data.intents import Example


def test_parse_banking77_csv_handles_quotes_and_commas() -> None:
    content = 'text,category\nwhere is it?,a_b\n"x, y, z",a_b\n'
    assert intents.parse_banking77_csv(content) == [
        Example("where is it?", "a_b"),
        Example("x, y, z", "a_b"),
    ]


def test_split_dev_is_disjoint_deterministic_and_per_intent() -> None:
    train = [Example(f"{lab} {i}", lab) for lab in ("a", "b") for i in range(10)]
    pool, dev = intents.split_dev(train, per_intent=3, seed=1)
    assert len(dev) == 6
    assert len(pool) == 14
    assert not set(pool) & set(dev)
    assert {ex.label for ex in dev} == {"a", "b"}
    assert intents.split_dev(train, per_intent=3, seed=1) == (pool, dev)


def test_examples_by_label_and_split() -> None:
    ds = tiny_dataset(n_labels=3, per_label=5)
    grouped = ds.examples_by_label()
    assert set(grouped) == set(ds.labels)
    assert all(len(v) == 5 for v in grouped.values())
    assert ds.split("dev") == ds.dev
    assert ds.split("test") == ds.test
    with pytest.raises(ValueError, match="unknown split"):
        ds.split("train")


def _synthetic_csvs() -> dict[str, bytes]:
    return {
        split: (
            "text,category\n"
            + "".join(f"{lab} {i},{lab}\n" for lab in ("x", "y") for i in range(rows))
        ).encode()
        for split, rows in (("train", 12), ("test", 2))
    }


def _pin(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    monkeypatch.setattr(
        intents,
        "BANKING77_SHA256",
        {split: hashlib.sha256(b).hexdigest() for split, b in files.items()},
    )


def test_load_banking77_uses_local_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _synthetic_csvs()
    _pin(monkeypatch, files)
    for split, body in files.items():
        dest = (
            tmp_path / "raw" / "banking77" / intents.BANKING77_COMMIT / f"{split}.csv"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    ds = intents.load("banking77", tmp_path)  # no network: files already present
    assert ds.labels == ("x", "y")
    assert len(ds.dev) == 2 * intents.DEV_PER_INTENT
    assert len(ds.pool) == 2 * (12 - intents.DEV_PER_INTENT)
    assert len(ds.test) == 4
    assert ds.oos_test == ()


def test_load_banking77_downloads_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _synthetic_csvs()
    _pin(monkeypatch, files)
    urls: list[str] = []

    def fetcher(url: str) -> bytes:
        urls.append(url)
        return files["train" if url.endswith("train.csv") else "test"]

    ds = intents.load_banking77(tmp_path, fetcher)
    again = intents.load_banking77(tmp_path, fetcher)
    assert len(urls) == 2
    assert all(intents.BANKING77_COMMIT in u for u in urls)
    assert ds == again
    raw = tmp_path / "raw" / "banking77" / intents.BANKING77_COMMIT
    assert sorted(p.name for p in raw.iterdir()) == ["test.csv", "train.csv"]


def test_load_banking77_rejects_a_tampered_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, _synthetic_csvs())
    with pytest.raises(IntegrityError, match="SHA-256"):
        intents.load_banking77(tmp_path, lambda _url: b"text,category\nevil,x\n")
    assert not (tmp_path / "raw").exists() or not any((tmp_path / "raw").rglob("*.csv"))


def test_pinned_banking77_digests_are_complete() -> None:
    assert set(intents.BANKING77_SHA256) == {"train", "test"}
    assert all(len(h) == 64 for h in intents.BANKING77_SHA256.values())


def test_load_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        intents.load("nope", tmp_path)


def test_select_items_is_stratified_and_nested() -> None:
    ds = tiny_dataset(n_labels=5)
    sel = intents.select_items(ds.dev, per_intent=3)
    assert Counter(ds.dev[i].label for i in sel) == dict.fromkeys(ds.labels, 3)
    sub = intents.nested_subset(ds.dev, sel, 1)
    assert set(sub) <= set(sel)
    assert Counter(ds.dev[i].label for i in sub) == dict.fromkeys(ds.labels, 1)
    assert intents.select_items(ds.dev, per_intent=3) == sel


def test_spread_indices() -> None:
    assert intents.spread_indices(20, 5) == [0, 4, 8, 12, 16]
    assert intents.spread_indices(3, 10) == [0]  # fewer items than requested
    ds = tiny_dataset(n_labels=5)
    idx = intents.spread_indices(len(ds.dev), 5)
    assert len({ds.dev[i].label for i in idx}) == 5


def test_filler_texts_reuses_loaded_clinc() -> None:
    ds = tiny_dataset(name="clinc150")
    assert intents.filler_texts({"clinc150": ds}, Path("unused")) == [
        ex.text for ex in ds.oos_pool
    ]
