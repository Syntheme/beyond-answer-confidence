import math

import numpy as np
import pytest

from beyond_answer_confidence.backends.base import ChoiceAnswer, Response
from beyond_answer_confidence.backends.cache import CallRecord
from beyond_answer_confidence.tasks.schema import Item
from beyond_answer_confidence.tasks.scoring import score_items


def rec(probs: dict[str, float]) -> CallRecord:
    return CallRecord(
        "k", Response("m", choices={"answer": ChoiceAnswer(probs)}), True, None, None
    )


def test_score_items_row_values() -> None:
    item = Item("u1", {}, options=("A", "B", "C"), gold="B")
    recs = [rec({"A": 0.2, "B": 0.8}), rec({"A": 0.4, "B": 0.4, "C": 0.2})]
    (row,) = score_items([item], {"u1": recs})
    assert row["top"] == "B"
    assert row["correct"] is True
    assert row["p_max"] == pytest.approx(0.6)
    p = np.array([0.3, 0.6, 0.1])
    assert row["norm_entropy"] == pytest.approx(-(p * np.log(p)).sum() / np.log(3))
    assert row["rep_tv"] == pytest.approx(0.4)


def test_single_option_has_no_normalised_entropy() -> None:
    item = Item("u1", {}, options=("A",), gold="A")
    (row,) = score_items([item], {"u1": [rec({"A": 1.0})]})
    assert row["norm_entropy"] is None
    assert row["p_max"] == 1.0
    assert not any(isinstance(v, float) and math.isnan(v) for v in row.values())


def test_all_zero_distribution_is_an_error() -> None:
    item = Item("u9", {}, options=("A", "B"), gold="A")
    with pytest.raises(ValueError, match="u9"):
        score_items([item], {"u9": [rec({"Z": 1.0})]})
