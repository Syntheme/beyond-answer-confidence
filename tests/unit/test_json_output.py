import json
import math
from pathlib import Path

import numpy as np
import pytest

from beyond_answer_confidence.experiments.base import (
    read_jsonl,
    write_json,
    write_jsonl,
)
from beyond_answer_confidence.json_output import dumps, finite


def test_finite_replaces_non_finite_floats_at_any_depth() -> None:
    obj = {
        "a": math.nan,
        "b": [1.0, math.inf, {"c": -math.inf, "d": (np.float64("nan"), 2)}],
        "e": "NaN",
        "f": True,
        "g": 3,
    }
    assert finite(obj) == {
        "a": None,
        "b": [1.0, None, {"c": None, "d": [None, 2]}],
        "e": "NaN",
        "f": True,
        "g": 3,
    }


def test_dumps_is_strict_and_keeps_finite_values_exact() -> None:
    x = 0.1 + 0.2
    text = dumps({"x": x, "y": np.float64(x), "z": math.nan})
    assert text == json.dumps({"x": x, "y": x, "z": None})
    # allow_nan cannot be switched back on, even for values made by `default`.
    with pytest.raises(ValueError, match="JSON compliant"):
        dumps({"x": object()}, allow_nan=True, default=lambda _o: math.nan)


def test_write_json_and_jsonl_never_emit_nan(tmp_path: Path) -> None:
    write_json(tmp_path / "s.json", {"m": math.nan, "n": [math.inf]})
    text = (tmp_path / "s.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"m": None, "n": [None]}
    write_jsonl(tmp_path / "r.jsonl", [{"p": -math.inf}, {"p": 0.5}])
    assert "Infinity" not in (tmp_path / "r.jsonl").read_text()
    assert read_jsonl(tmp_path / "r.jsonl") == [{"p": None}, {"p": 0.5}]
