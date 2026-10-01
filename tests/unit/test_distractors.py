import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from beyond_answer_confidence.data import distractors as dis

# SHA-256 of public inputs (not a secret).
PINNED_INDEX_KEY = "f4bf047d1d8cf57cb94d85f9ab321b3ee44721211cb9ecb84d2016283ed98cd2"  # pragma: allowlist secret


def test_normalise_answer() -> None:
    assert dis.normalise_answer("  Paris, France! ") == "paris france"
    assert dis.normalise_answer("...") == ""


def test_sample_distractors_skips_aliases_and_duplicates() -> None:
    rng = random.Random(0)  # noqa: S311
    d = dis.sample_distractors(
        "Paris", ["paris", "Lyon", "Nice", "Lille", "Lyon"], ["Paris, France"], 3, rng
    )
    assert d is not None
    assert sorted(d) == ["Lille", "Lyon", "Nice"]
    assert dis.sample_distractors("Paris", ["Lyon", "Nice"], [], 3, rng) is None


def test_perturb_answer_years_and_numbers() -> None:
    rng = random.Random(0)  # noqa: S311
    years = dis.perturb_answer("12 May 1998", rng)
    assert years is not None
    assert len(set(years)) == 3
    assert all(y != "12 May 1998" and y.startswith("12 May ") for y in years)
    assert all(abs(int(y[-4:]) - 1998) <= 6 for y in years)
    nums = dis.perturb_answer("120,000 euros", rng)
    assert nums is not None
    assert all(n.endswith(" euros") and n != "120,000 euros" and "," in n for n in nums)
    decimals = dis.perturb_answer("3.25 m", rng)
    assert decimals is not None
    assert all(len(d.split(" ")[0].split(".")[1]) == 2 for d in decimals)
    assert dis.perturb_answer("no digits", rng) is None
    # Scaling 1 by 0.7, 0.8 or 1.25 rounds to 2 each time: not distinct.
    assert dis.perturb_answer("1", random.Random(0), k=7) is None  # noqa: S311


def test_index_key_format_is_fixed() -> None:
    # Saved distractor choices are looked up by this digest.
    key = dis.index_key(["Paris"], [["Paree"]], ["Lyon", "Berlin", "Zürich"], 3, 0.92)
    assert key == PINNED_INDEX_KEY


POOL = ["Paris", "Lyon", "Berlin", "Munich", "Rome", "Milan", "Madrid"]


def fake_embed(texts: Sequence[str], _d: Path, batch: int = 256) -> np.ndarray:
    vec = {w: np.eye(len(POOL))[i] + 0.1 for i, w in enumerate(POOL)}
    m = np.array([vec.get(t, np.ones(len(POOL))) for t in texts], float)
    out: np.ndarray = m / np.linalg.norm(m, axis=1, keepdims=True)
    return out


def test_nearest_distractors_are_saved_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dis, "embed", fake_embed)
    first = dis.nearest_distractors(["Paris"], [[]], POOL, tmp_path, k=3)
    saved = list((tmp_path / "cache" / "distractors").glob("*.json"))
    assert len(saved) == 1
    body = json.loads(saved[0].read_text())
    assert all(isinstance(j, int) for j in body["picks"][0])
    assert "Paris" not in json.dumps(body)  # positions only, no text

    def broken(*_a: object, **_k: object) -> np.ndarray:
        raise AssertionError("embeddings must not be recomputed")

    monkeypatch.setattr(dis, "embed", broken)
    assert dis.nearest_distractors(["Paris"], [[]], POOL, tmp_path, k=3) == first
    assert first[0] is not None
    assert "Paris" not in first[0]
    assert len(set(first[0])) == 3


def test_nearest_indices_known_answer() -> None:
    uniq = ["alpha", "alphabet", "beta", "gamma", "delta"]
    pool_vec = np.array(
        [[1, 0, 0], [0.99, 0.141, 0], [0.8, 0.6, 0], [0.6, 0.8, 0], [0, 0, 1]], float
    )
    pool_vec /= np.linalg.norm(pool_vec, axis=1, keepdims=True)
    gold_vec = np.array([[1.0, 0.0, 0.0]])
    # "alpha" is the gold, "alphabet" contains it, so the nearest others win.
    picks = dis.nearest_indices(["alpha"], [[]], uniq, pool_vec, gold_vec, 2, 0.999)
    assert picks == [[2, 3]]
    assert dis.nearest_indices(["alpha"], [[]], uniq, pool_vec, gold_vec, 4, 0.999) == [
        None
    ]


def test_distractor_picker_binds_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dis, "embed", fake_embed)
    pick = dis.distractor_picker(tmp_path, k=2)
    out = pick(["Rome"], [[]], POOL)
    assert out[0] is not None
    assert len(out[0]) == 2
    assert (tmp_path / "cache" / "distractors").is_dir()


def test_embed_mean_pools_and_normalises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    seen: dict[str, object] = {}

    class Tok:
        def __call__(self, texts: list[str], **kw: object) -> dict[str, object]:
            n = max(len(t) for t in texts)
            mask = torch.tensor([[1] * len(t) + [0] * (n - len(t)) for t in texts])
            return {"input_ids": mask.clone(), "attention_mask": mask}

    class Out:
        def __init__(self, hidden: object) -> None:
            self.last_hidden_state = hidden

    class Model:
        def eval(self) -> None:
            seen["eval"] = True

        def __call__(self, input_ids: object, attention_mask: object) -> Out:
            ids = torch.as_tensor(input_ids).float()
            return Out(torch.stack([ids, 2 * ids, torch.ones_like(ids)], dim=-1))

    def tok_loader(name: str, **kw: object) -> Tok:
        seen["tok"] = (name, kw)
        return Tok()

    def model_loader(name: str, **kw: object) -> Model:
        seen["model"] = (name, kw)
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tok_loader)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model_loader)
    vec = dis.embed(["ab", "abcd", "a"], tmp_path, batch=2)
    assert vec.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1.0)
    assert np.allclose(vec[0], vec[1])  # same direction after mean pooling
    assert seen["eval"] is True
    assert seen["model"] == (
        dis.ENCODER,
        {
            "revision": dis.ENCODER_REVISION,
            "cache_dir": str(tmp_path / "hf"),
            "use_safetensors": True,
        },
    )
    assert dis.embed([], tmp_path).shape == (0, 384)
