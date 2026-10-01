import gzip
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from beyond_answer_confidence import export
from beyond_answer_confidence.export import UnsafeExportError


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")


SAFE = [
    {"unit": "a:1", "gold": "A", "dist": {"A": 0.7, "B": 0.3}, "p_max": 0.7},
    {"unit": "a:2", "gold": "B", "dist": {"A": 0.4, "B": 0.6}, "p_max": 0.6},
]


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    write_rows(root / "exp" / "rows.jsonl", SAFE)
    (root / "exp" / "summary.json").write_text(
        json.dumps({"results": {"x": 1.0, "_draws": [1, 2]}, "list": [{"_p": 1}]})
    )
    (root / "exp" / "run.log").write_text("not exported")
    np.savez(
        root / "exp" / "pred.npz",
        probs=np.ones((2, 2)),
        labels=np.array(["a", "b"]),
    )
    write_rows(root / "other" / "rows.jsonl", SAFE)
    return root


def test_strip_private() -> None:
    obj = {"a": 1, "_b": 2, "c": [{"_d": 3, "e": 4}]}
    assert export.strip_private(obj) == {"a": 1, "c": [{"e": 4}]}


def test_check_rows_counts(tmp_path: Path) -> None:
    write_rows(tmp_path / "r.jsonl", SAFE)
    assert export.check_rows(tmp_path / "r.jsonl") == 2


@pytest.mark.parametrize(
    "row",
    [
        {"unit": "x", "question": "q"},
        {"unit": "x", "state": {}},
        {"unit": "x", "text": "t"},
        {"unit": "x", "gold": "y" * 101},
        {"unit": "x", "dist": {"k" * 101: 1.0}},
        {"unit": "x", "nested": [{"deep": "z" * 200}]},
    ],
)
def test_check_rows_refuses_text(tmp_path: Path, row: dict[str, object]) -> None:
    write_rows(tmp_path / "r.jsonl", [SAFE[0], row])
    with pytest.raises(UnsafeExportError):
        export.check_rows(tmp_path / "r.jsonl")


@pytest.mark.parametrize(
    "row",
    [
        {"unit": "x", "meta": {"question": "q"}},
        {"unit": "x", "cells": [{"a": 1}, {"deep": {"sentence": ["s"]}}]},
        {"unit": "x", "info": {"state": {"inner": "t"}}},
    ],
)
def test_check_rows_refuses_nested_short_text(
    tmp_path: Path, row: dict[str, object]
) -> None:
    write_rows(tmp_path / "r.jsonl", [row])
    with pytest.raises(UnsafeExportError, match="text field"):
        export.check_rows(tmp_path / "r.jsonl")


def test_nested_text_named_labels_with_numbers_are_allowed(tmp_path: Path) -> None:
    # e.g. an intent label called "text" mapping to its probability
    row: dict[str, object] = {
        "unit": "x",
        "probs": {"text": 0.2, "other": 0.8},
        "n": {"original": 3},
    }
    write_rows(tmp_path / "r.jsonl", [row])
    assert export.check_rows(tmp_path / "r.jsonl") == 1


def test_check_rows_refuses_non_objects(tmp_path: Path) -> None:
    (tmp_path / "r.jsonl").write_text("[1, 2]\n")
    with pytest.raises(UnsafeExportError, match="not an object"):
        export.check_rows(tmp_path / "r.jsonl")


def test_check_npz(tmp_path: Path) -> None:
    np.savez(tmp_path / "ok.npz", a=np.arange(3), b=np.array(["x", "y"]))
    export.check_npz(tmp_path / "ok.npz")
    np.savez(tmp_path / "long.npz", b=np.array(["x" * 500]))
    with pytest.raises(UnsafeExportError, match="long strings"):
        export.check_npz(tmp_path / "long.npz")
    np.savez(tmp_path / "obj.npz", b=np.array([{"a": 1}], dtype=object))
    with pytest.raises(UnsafeExportError):
        export.check_npz(tmp_path / "obj.npz")
    np.savez(tmp_path / "bytes.npz", b=np.array([b"x"]))
    with pytest.raises(UnsafeExportError, match="dtype"):
        export.check_npz(tmp_path / "bytes.npz")


def test_build_writes_manifest_and_notes(outputs: Path, tmp_path: Path) -> None:
    dis = tmp_path / "distractors"
    dis.mkdir()
    (dis / "set.json").write_text(json.dumps({"q1": [3, 1]}))
    out = tmp_path / "export"
    manifest = export.build(outputs, ["exp", "missing"], out, distractors=dis)
    paths = {f["path"] for f in manifest["files"]}
    assert paths == {
        "exp/rows.jsonl.gz",
        "exp/summary.json",
        "exp/pred.npz",
        "distractors/set.json",
    }
    rows_entry = next(f for f in manifest["files"] if f["path"].endswith(".gz"))
    assert rows_entry["rows"] == 2
    for f in manifest["files"]:
        assert f["sha256"] == export.sha256(out / f["path"])
    with gzip.open(out / "exp/rows.jsonl.gz", "rt") as fh:
        assert json.loads(fh.readline())["unit"] == "a:1"
    summary = json.loads((out / "exp/summary.json").read_text())
    assert summary == {"results": {"x": 1.0}, "list": [{}]}
    readme = (out / "README.md").read_text()
    assert "No dataset text" in readme
    datasets = (out / "DATASETS.md").read_text()
    assert "ChaosNLI" in datasets
    assert "non-commercial" in datasets
    assert json.loads((out / "MANIFEST.json").read_text())["experiments"] == [
        "exp",
        "missing",
    ]
    assert json.loads((out / "MANIFEST.json").read_text())["package"] == (
        "beyond-answer-confidence"
    )
    # A previous export is replaced.
    export.build(outputs, ["other"], out)
    assert not (out / "exp").exists()
    tar = export.make_tar(out)
    with tarfile.open(tar) as t:
        assert "export/MANIFEST.json" in t.getnames()
    # An existing archive is kept unless forced.
    with pytest.raises(FileExistsError, match="--force"):
        export.make_tar(out)
    assert export.make_tar(out, force=True) == tar


def test_build_refuses_unsafe_rows_without_writing(
    outputs: Path, tmp_path: Path
) -> None:
    write_rows(outputs / "exp" / "bad.jsonl", [{"unit": "x", "premise": "p"}])
    out = tmp_path / "export"
    with pytest.raises(UnsafeExportError, match="premise"):
        export.build(outputs, ["exp"], out)
    assert not out.exists()


def test_build_refuses_long_strings_in_results(outputs: Path, tmp_path: Path) -> None:
    (outputs / "exp" / "summary.json").write_text(json.dumps({"x": "y" * 300}))
    with pytest.raises(UnsafeExportError):
        export.build(outputs, ["exp"], tmp_path / "export")


def test_build_refuses_nested_text_in_results(outputs: Path, tmp_path: Path) -> None:
    (outputs / "exp" / "summary.json").write_text(
        json.dumps({"results": {"examples": [{"prompt": "short"}]}})
    )
    with pytest.raises(UnsafeExportError, match="prompt"):
        export.build(outputs, ["exp"], tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_build_refuses_text_fields_in_distractors(
    outputs: Path, tmp_path: Path
) -> None:
    dis = tmp_path / "distractors"
    dis.mkdir()
    (dis / "set.json").write_text(json.dumps({"picks": [[1]], "text": "s"}))
    with pytest.raises(UnsafeExportError, match="text field"):
        export.build(outputs, ["exp"], tmp_path / "export", distractors=dis)


def test_export_json_is_strict(outputs: Path, tmp_path: Path) -> None:
    (outputs / "exp" / "summary.json").write_text('{"a": NaN, "b": [Infinity, 1.5]}')
    write_rows(outputs / "exp" / "rows.jsonl", [{"unit": "u", "p": float("nan")}])
    out = tmp_path / "export"
    export.build(outputs, ["exp"], out)
    text = (out / "exp/summary.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None, 1.5]}
    with gzip.open(out / "exp/rows.jsonl.gz", "rt") as fh:
        body = fh.read()
    assert "NaN" not in body
    assert json.loads(body) == {"unit": "u", "p": None}


def test_build_refuses_unsafe_distractors(outputs: Path, tmp_path: Path) -> None:
    dis = tmp_path / "distractors"
    dis.mkdir()
    (dis / "set.json").write_text(json.dumps({"q": "x" * 300}))
    with pytest.raises(UnsafeExportError):
        export.build(outputs, ["exp"], tmp_path / "export", distractors=dis)


def test_build_refuses_foreign_or_nested_targets(outputs: Path, tmp_path: Path) -> None:
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export.build(outputs, ["exp"], foreign)
    assert (foreign / "keep.txt").exists()
    plain_file = tmp_path / "file"
    plain_file.write_text("x")
    with pytest.raises(FileExistsError):
        export.build(outputs, ["exp"], plain_file)
    with pytest.raises(ValueError, match="inside"):
        export.build(outputs, ["exp"], outputs / "exp" / "export")


def test_build_replaces_only_its_own_exports(outputs: Path, tmp_path: Path) -> None:
    # A directory with some other MANIFEST.json is not a previous export.
    other = tmp_path / "other"
    other.mkdir()
    (other / "MANIFEST.json").write_text(json.dumps({"files": []}))
    (other / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="not a previous"):
        export.build(outputs, ["exp"], other)
    assert (other / "keep.txt").exists()
    (other / "MANIFEST.json").write_text("not json")
    with pytest.raises(FileExistsError):
        export.build(outputs, ["exp"], other)
    empty = tmp_path / "empty"
    empty.mkdir()
    export.build(outputs, ["exp"], empty)
    assert export.is_previous_export(empty)


def test_dataset_notes_are_complete() -> None:
    for d in export.DATASETS:
        assert d.name
        assert d.source
        assert d.revision
        assert d.licence
