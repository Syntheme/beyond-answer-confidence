import hashlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from beyond_answer_confidence.data import download, loaders, mirrors


def test_hf_file_pins_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(repo: str, filename: str, **kw: Any) -> str:
        calls.append({"repo": repo, "filename": filename, **kw})
        return str(tmp_path / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    path = loaders.hf_file("earino/chaosnli", "raw/x.jsonl", tmp_path)
    assert path == tmp_path / "raw/x.jsonl"
    assert calls[0]["revision"] == loaders.REVISIONS["earino/chaosnli"]
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["cache_dir"] == str(tmp_path / "hf")
    with pytest.raises(KeyError):
        loaders.hf_file("someone/unpinned", "f", tmp_path)


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Serve synthetic files in place of the Hub."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "test.tsv").write_text(
        "id\tsubj\tprop\tobj\tquestion\tpossible_answers\to_aliases\ts_pop\n"
        '1\tThing\tgenre\tjazz\tWhat genre is Thing?\t["jazz"]\t\t5\n'
    )
    (src / "tf_questions_2020-01-01_2026-07-18.csv").write_text(
        "question,answer,date,category\nWill it?,Yes,2024-03-05,misc\n"
    )
    (src / "SelfAware.json").write_text(
        json.dumps(
            {"example": [{"question_id": 1, "question": "q", "answerable": True}]}
        )
    )
    (src / "simpleqa_verified.csv").write_text("problem,answer\nq,NA\n")
    (src / "chaos.jsonl").write_text(
        json.dumps({"uid": "u", "label_count": [1, 2, 3]}) + "\n"
    )
    pd.DataFrame({"a": [1, 2]}).to_parquet(src / "table.parquet")

    def fake_hf_file(repo: str, filename: str, data_dir: Path) -> Path:
        assert repo in loaders.REVISIONS
        name = Path(filename).name
        if name.endswith(".parquet"):
            name = "table.parquet"
        if name.startswith("chaosNLI_"):
            name = "chaos.jsonl"
        return src / name

    monkeypatch.setattr(loaders, "hf_file", fake_hf_file)
    return tmp_path


def test_loaders_parse_synthetic_files(files: Path) -> None:
    popqa = loaders.load_popqa(files)
    assert popqa.loc[0, "possible_answers"] == ["jazz"]
    assert popqa.loc[0, "o_aliases"] == ""
    assert loaders.load_daily_oracle("tf", files).loc[0, "month"] == "2024-03"
    assert loaders.load_selfaware(files).loc[0, "question_id"] == 1
    assert loaders.load_simpleqa(files).loc[0, "answer"] == "NA"  # not NaN
    assert list(loaders.load_chaosnli("snli", files)["uid"]) == ["u"]
    for load in (
        loaders.load_hotpot_validation,
        loaders.load_truthfulqa_mc1,
        loaders.load_triviaqa_validation,
        loaders.load_ambigqa_validation,
    ):
        assert list(load(files)["a"]) == [1, 2]
    assert list(loaders.load_qanta("guessdev", files)["a"]) == [1, 2]


def test_truthfulqa_csv_is_downloaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"Question,Best Answer\nq,a\n"
    monkeypatch.setattr(
        loaders, "TRUTHFULQA_CSV_SHA256", hashlib.sha256(body).hexdigest()
    )
    urls: list[str] = []

    def fetcher(url: str) -> bytes:
        urls.append(url)
        return body

    first = loaders.load_truthfulqa_binary(tmp_path, fetcher)
    second = loaders.load_truthfulqa_binary(tmp_path, fetcher)
    assert len(urls) == 1
    assert loaders.TRUTHFULQA_CSV_COMMIT in urls[0]
    assert urls[0].startswith("https://")
    assert first.equals(second)
    assert list(first.columns) == ["Question", "Best Answer"]
    assert [p.name for p in (tmp_path / "raw").iterdir()] == [
        f"TruthfulQA_{loaders.TRUTHFULQA_CSV_COMMIT[:12]}.csv"
    ]


def test_truthfulqa_csv_with_wrong_digest_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(download.IntegrityError, match="nothing was written"):
        loaders.load_truthfulqa_binary(tmp_path, lambda _url: b"Question\nforged\n")
    assert not (tmp_path / "raw").exists() or not any((tmp_path / "raw").iterdir())


def test_pinned_file_rejects_a_modified_local_copy(tmp_path: Path) -> None:
    dest = tmp_path / "f.csv"
    good = b"a,b\n"
    digest = hashlib.sha256(good).hexdigest()
    assert download.pinned_file("https://x/f", dest, digest, lambda _u: good) == dest
    assert dest.read_bytes() == good
    dest.write_bytes(b"a,b\nextra\n")
    with pytest.raises(download.IntegrityError, match="delete the file"):
        download.pinned_file("https://x/f", dest, digest, lambda _u: good)


def test_pinned_file_leaves_no_partial_file_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = b"payload"
    digest = hashlib.sha256(good).hexdigest()

    def broken_replace(*_a: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        download.pinned_file("https://x/f", tmp_path / "f", digest, lambda _u: good)
    assert list(tmp_path.iterdir()) == []


def test_to_jsonable_and_table_rows() -> None:
    value = {"a": np.array([1, 2]), "b": [np.float64(0.5)], "c": np.int64(3)}
    assert loaders.to_jsonable(value) == {"a": [1, 2], "b": [0.5], "c": 3}
    rows = list(loaders.table_rows(pd.DataFrame({"x": [1, 2]}, index=[5, 6])))
    assert [(r.Index, r.x) for r in rows] == [(5, 1), (6, 2)]


def test_mirror_comparisons() -> None:
    a = [{"question_id": 1, "q": "x"}, {"question_id": 2, "q": "y"}]
    b = [
        {"question_id": 1, "q": "x"},
        {"question_id": 2, "q": "z"},
        {"question_id": 3, "q": "w"},
    ]
    res = mirrors.compare_selfaware(a, b)
    assert (res["differing"], res["only_in_original"], res["shared"]) == (1, 1, 2)
    assert res["differing_examples"] == ["2"]
    m = [
        {
            "uid": "u",
            "label_count": [1, 2, 3],
            "old_labels": ["e"],
            "example": {"premise": "p", "hypothesis": "h"},
        }
    ]
    o = [
        {
            "uid": "u",
            "label_count": [1, 2, 4],
            "old_labels": ["e"],
            "premise": "p",
            "hypothesis": "h2",
        }
    ]
    res2 = mirrors.compare_chaos_mnli(m, o)
    assert (
        res2["label_counts_differ"],
        res2["old_labels_differ"],
        res2["text_differs"],
    ) == (1, 0, 1)


def test_check_mirrors_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sa = [{"question_id": 1, "question": "q", "answerable": True}]
    mnli = [
        {
            "uid": "u",
            "label_count": [1, 2, 3],
            "old_labels": ["e"],
            "example": {"premise": "p", "hypothesis": "h"},
        }
    ]
    counts = dict(mirrors.PUBLISHED_COUNTS)

    def fake_chaos(subset: str, _d: Path) -> pd.DataFrame:
        if subset == "mnli_m":
            return pd.DataFrame(mnli * counts["mnli_m"])
        return pd.DataFrame({"uid": range(counts[subset])})

    monkeypatch.setattr(loaders, "load_selfaware", lambda _d: pd.DataFrame(sa))
    monkeypatch.setattr(loaders, "load_chaosnli", fake_chaos)
    served = {
        mirrors.SELFAWARE_GITHUB: json.dumps({"example": sa}).encode(),
        mirrors.TASKSOURCE_MNLI: (
            json.dumps(
                {
                    "uid": "u",
                    "label_count": [1, 2, 3],
                    "old_labels": ["e"],
                    "premise": "p",
                    "hypothesis": "h",
                }
            )
            + "\n\n"
        ).encode(),
    }
    res = mirrors.check_mirrors(tmp_path, served.__getitem__)
    assert res["consistent"] is True
    assert res["selfaware"]["shared"] == 1
    assert res["chaosnli_mnli"]["shared"] == 1
    assert res["chaosnli_counts_vs_published"]["snli"] == {
        "mirror": 1514,
        "published": 1514,
    }
    counts["snli"] = 10
    assert mirrors.check_mirrors(tmp_path, served.__getitem__)["consistent"] is False


def test_fetch_refuses_plain_http() -> None:
    with pytest.raises(ValueError, match="https"):
        download.fetch("http://example.com/file")


def test_fetch_reads_https(monkeypatch: pytest.MonkeyPatch) -> None:
    class Body(io.BytesIO):
        def __enter__(self) -> "Body":
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Body(b"ok"))
    assert download.fetch("https://example.com/file") == b"ok"
